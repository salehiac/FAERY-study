# Novelty Search Library.
# Copyright (C) 2020 Sorbonne University
# Maintainer: Achkan Salehi (salehi@isir.upmc.fr)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import time
import os
import copy
import random
import gc
import torch

from scoop import futures
from termcolor import colored

import utils_misc


class NoveltySearch:

    BD_VIS_DISABLE = 0
    BD_VIS_TO_FILE = 1
    BD_VIS_DISPLAY = 2

    def __init__(
            self,
            archive,
            nov_estimator,
            mutator,
            problem,
            selector,
            n_pop,
            n_offspring,
            agent_factory,
            visualise_bds_flag,
            map_type="scoop",
            logs_root="/tmp/ns_log/",
            compute_parent_child_stats=False,
            initial_pop=[],  # make sure they are passed by deepcopy
            problem_sampler=None):
        """
        archive                      Archive           object implementing the Archive interface. Can be None if novelty is LearnedNovelty1d/LearnedNovelty2d
        nov_estimator                NoveltyEstimator  object implementing the NoveltyEstimator interface. 
        problem                      Problem           None, or object that provides 
                                                            - __call__ function taking individual_index returning (fitness, behavior_descriptors, task_solved_or_not)
                                                            - a dist_thresh (that is determined from its bds) which specifies the minimum distance that should separate a point x from
                                                              its nearest neighbour in the archive+pop in order for the point to be considered as novel. It is also used as a threshold on novelty
                                                              when updating the archive.
                                                           - optionally, a visualise_bds function.
                                                        When problem_sampler is not None, problem should be set to None.
        mutator                      Mutator
        selector                     function
        n_pop                        int 
        n_offspring                  int           
        agent_factory                function          used to 1. create intial population and 2. convert mutated list genotypes back to agent types.
                                                       Currently, considered agents should 
                                                           - inherit from list (to avoid issues with deap functions that have trouble with numpy slices)
                                                           - provide those fields: _fitness, _behavior_descr, _novelty, _solved_task, _created_at_gen
                                                       This is just to facilitate interactions with the deap library
        visualise_bds_flag           int               gets a visualisation of the behavior descriptors (assuming the Problem and its descriptors allow it), and either display it or save it to 
                                                       the logs dir.
                                                       possible values are BD_VIS_TO_FILE and BD_VIS_DISPLAY
        map_type                     string            different options for sequential/parallel mapping functions. supported values currently are 
                                                       "scoop" distributed map from futures.map
                                                       "std"   buildin python map
        logs_root                    str               the logs diretory will be created inside logs_root
        compute_parent_child_stats   bool
        initial_pop                  lst               if a prior on the population exists, it should be supplied here. ATTENTION: it should NOT be passed by ref
        problem_sampler              function          problem_sampler(num_samples=n) should return n instances of the problem 
        """
        self.archive = archive
        if archive is not None:
            self.archive.reset()

        self.nov_estimator = nov_estimator
        self.problem = problem

        if problem_sampler is not None:
            assert self.problem is None, "you have provided a problem sampler and a problem. Please decide which you want, and set the other to None."
            self.problem = problem_sampler(num_samples=1)[0]

        self.map_type = map_type
        self._map = futures.map if map_type == "scoop" else map
        #print(colored("[NS info] Using map_type "+map_type, "green",attrs=["bold"]))

        self.mutator = mutator
        self.selector = selector

        self.n_offspring = n_offspring
        self.agent_factory = agent_factory

        # this is important for attributing _idx values to future agents
        self.num_agent_instances = n_pop

        if not len(initial_pop):
            print(
                colored(
                    "[NS info] No initial population prior, initialising from scratch",
                    "magenta",
                    attrs=["bold"]))
            initial_pop = [self.agent_factory(i) for i in range(n_pop)]
            initial_pop = self.generate_new_agents(initial_pop, generation=0)
        else:
            assert len(initial_pop) == n_pop, " this shouldn't happen"
            for x in initial_pop:
                x._created_at_gen = 0

        self._initial_pop = copy.deepcopy(initial_pop)

        assert n_offspring >= len(
            initial_pop), "n_offspring should be larger or equal to n_pop"

        # self.visualise_bds_flag = visualise_bds_flag
        self.visualise_bds_flag = NoveltySearch.BD_VIS_DISABLE
        
        if os.path.isdir(logs_root):
            self.logs_root = logs_root
            self.log_dir_path = utils_misc.create_directory_with_pid(
                dir_basename=logs_root + "/NS_log_" + utils_misc.rand_string() +
                "_",
                remove_if_exists=True,
                no_pid=False)
        else:
            raise Exception(
                "Root dir for logs not found. Please ensure that it exists before launching the script."
            )

        # for problems for which it is relevant (e.g. hardmaze), keep track of individuals that have solved the task
        self.task_solvers = {}  # key,value=generation, list(agents)

        self.compute_parent_child_stats = compute_parent_child_stats

        self.save_archive_to_file = False
        self.disable_tqdm = False

    def eval_agents(self, agents):
        tt1 = time.time()
        xx = list(
            self._map(self.problem, agents)
        )  # attention, don't deepcopy the problem instance. Use problem_sampler if necessary
        tt2 = time.time()
        elapsed = tt2 - tt1
        task_solvers = []
        for ag_i in range(len(agents)):
            ag = agents[ag_i]
            ag._fitness = xx[ag_i][0]
            ag._behavior_descr = xx[ag_i][1]
            ag._solved_task = xx[ag_i][2]
            # for debug only, disable it later
            ag._complete_trajs = xx[ag_i][3]
            ag._last_eval_init_state = xx[ag_i][4]
            ag._last_eval_init_obs = xx[ag_i][5]
            ag._first_action = xx[ag_i][6]

            if hasattr(self.problem, "get_task_info"):
                ag._task_info = self.problem.get_task_info()

            if ag._solved_task:
                task_solvers.append(ag)

        return task_solvers, elapsed

    def __call__(self,
                 iters,
                 stop_on_reaching_task=True,
                 reinit=False,
                 save_checkpoints=0):
        """
        iters  int  number of iterations
        """
        with torch.no_grad():
            print(
                f"Starting NS with pop_sz={len(self._initial_pop)}, offspring_sz={self.n_offspring}",
                flush=True)

            if save_checkpoints:
                raise NotImplementedError(
                    "checkpoint save/load not implemented yet")

            if reinit and self.archive is not None:
                self.archive.reset()

            parents = copy.deepcopy(
                self._initial_pop
            )  # pop is a member in order to avoid passing copies to workers
            self.eval_agents(parents)

            self.nov_estimator.update(archive=[], pop=parents)
            novs = self.nov_estimator()  # computes novelty of all population
            for ag_i in range(len(parents)):
                parents[ag_i]._nov = novs[ag_i]

            #tqdm_gen = tqdm.trange(iters, desc='', leave=True, disable=self.disable_tqdm)
            for it in range(iters):
                print("Inner g : {}/{}".format(it, iters), end="\r")

                offsprings = self.generate_new_agents(
                    parents, generation=it + 1
                )  # mutations and crossover happen here  <<= deap can be useful here
                task_solvers, _ = self.eval_agents(offsprings)

                pop = parents + offsprings  # all of them have _fitness and _behavior_descr now

                for x in pop:
                    if x._age == -1:
                        x._age = it + 1 - x._created_at_gen
                    else:
                        x._age += 1

                self.nov_estimator.update(archive=self.archive, pop=pop)
                novs = self.nov_estimator(
                )  # computes novelty of all population
                for ag_i in range(len(pop)):
                    pop[ag_i]._nov = novs[ag_i]

                parents_next = self.selector(individuals=pop)

                if self.compute_parent_child_stats:
                    for x in parents_next:
                        # otherwise it has already been computed in a previous generation
                        if x._bd_dist_to_parent_bd == -1 and x._created_at_gen > 0:
                            xp = next(
                                (s for s in pop if s._idx == x._parent_idx),
                                None)
                            if xp is None:
                                raise Exception("this shouldn't happen")
                            x._bd_dist_to_parent_bd = self.problem.bd_extractor.distance(
                                x._behavior_descr, xp._behavior_descr)

                parents = parents_next
                if hasattr(self.nov_estimator, "train"):
                    self.nov_estimator.train(parents)
                if self.archive is not None:
                    self.archive.update(parents,
                                        offsprings,
                                        thresh=self.problem.dist_thresh,
                                        boundaries=[0, 600],
                                        knn_k=15)
                    if self.save_archive_to_file:
                        self.archive.dump(self.log_dir_path + f"/archive_{it}")

                if len(task_solvers):
                    self.visualise_bds(
                        parents + [x for x in offsprings if x._solved_task],
                        generation_num=it)
                    utils_misc.dump_pickle(
                        self.log_dir_path + f"/population_gen_{it}", parents)

                if len(task_solvers):
                    print(colored("[NS info] found task solvers (generation " +
                                  str(it) + ")",
                                  "magenta",
                                  attrs=["bold"]),
                          flush=True)
                    self.task_solvers[it] = task_solvers
                    if stop_on_reaching_task:
                        break
                gc.collect()

                #tqdm_gen.set_description(f"Generation {it}/{iters}, archive_size=={len(self.archive) if self.archive is not None else -1}")
                # tqdm_gen.refresh()

            return parents, self.task_solvers  # iteration:list_of_agents

    def generate_new_agents(self, parents, generation: int):

        parents_as_list = [(x._idx, x.get_flattened_weights(), x._root)
                           for x in parents]
        parents_to_mutate = random.choices(
            range(len(parents_as_list)),
            k=self.n_offspring)  # note that usually n_offspring>=len(parents)
        mutated_genotype = [
            (parents_as_list[i][0],
             self.mutator(copy.deepcopy(parents_as_list[i][1])),
             parents_as_list[i][2]) for i in parents_to_mutate
        ]  # deepcopy is because of deap

        num_s = self.n_offspring if generation != 0 else len(parents_as_list)

        mutated_ags = [
            self.agent_factory(self.num_agent_instances + x)
            for x in range(num_s)
        ]
        kept = random.sample(range(len(mutated_genotype)), k=num_s)
        for i in range(len(kept)):
            mutated_ags[i]._parent_idx = mutated_genotype[kept[i]][0]
            mutated_ags[i].set_flattened_weights(
                mutated_genotype[kept[i]][1][0])
            mutated_ags[i]._created_at_gen = generation
            mutated_ags[i]._root = mutated_genotype[kept[i]][2]

        self.num_agent_instances += len(mutated_ags)

        for x in mutated_ags:
            x.eval()

        return mutated_ags

    def visualise_bds(self, agents, generation_num=-1):

        if self.visualise_bds_flag != NoveltySearch.BD_VIS_DISABLE:  # and it%10==0:
            q_flag = True if self.visualise_bds_flag == NoveltySearch.BD_VIS_TO_FILE else False
            archive_it = iter(self.archive) if self.archive is not None else []
            self.problem.visualise_bds(archive_it,
                                       agents,
                                       quitely=q_flag,
                                       save_to=self.log_dir_path,
                                       generation_num=generation_num)