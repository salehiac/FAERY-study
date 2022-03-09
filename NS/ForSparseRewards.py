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

from curses import meta
import os
import pickle
import metaworld
import numpy as np

from abc import ABC, abstractmethod
from termcolor import colored
from scoop import futures

import NS
import Archives
import NoveltyEstimators
import MiscUtils
import MetaworldProblems

from utils_sparse_rewards import _mutate_initial_prior_pop, _mutate_prior_pop, ns_instance


class ForSparseRewards(ABC):
    """
    Base skeleton for the implemented meta-algorithms
    """

    def __init__(self,
                 pop_sz,
                 off_sz,
                 G_outer,
                 G_inner,
                 train_sampler,
                 test_sampler,
                 num_train_samples,
                 num_test_samples,
                 agent_factory,
                 top_level_log_root="tmp/mqd_tmp/",
                 resume_from_gen={}):
        """
        Note: unlike many meta algorithms, the improvements of the outer loop are not based on validation data, but on meta observations from the inner loop. So the
        test_sampler here is used for the evaluation of the meta algorithm, not for learning, i.e. no sample-splitting
        pop_sz               int          population size
        off_sz               int          number of offsprings
        G_outer              int          number of generations in the outer (meta) loop
        G_inner              int          number of generations in the inner loop (i.e. num generations for each QD problem)
        train_sampler        functor      any object/functor/function with a __call__ that returns a list of problems from a training distribution of environments
        test_sampler         function     any object/functor/function with a __call__ that returns a list of problems from a test distribution of environments
        num_train_samples    int          number of environments to use at each outer_loop generation for training
        num_test_samples     int          number of environments to use at each outer_loop generation for testing
        agent_factory        function     either _make_2d_maze_ag or _make_metaworld_ml1_ag
        top_level_log_root   str          where to save the population after each top_level optimisation
        resume_from_gen      dict         If not empty, then should be of the form {"gen":some_int, "init_pop":[list of agents]} such that the agent match with agent_factory
        """

        self.pop_sz = pop_sz
        self.off_sz = off_sz
        self.G_outer = G_outer
        self.G_inner = G_inner
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.agent_factory = agent_factory
        self.resume_from_gen = resume_from_gen

        if os.path.isdir(top_level_log_root):
            self.top_level_log = MiscUtils.create_directory_with_pid(
                dir_basename=top_level_log_root + "/meta-learning_" +
                MiscUtils.rand_string() + "_",
                remove_if_exists=True,
                no_pid=False)
            print(
                colored(
                    "[NS info] temporary dir for meta-learning was created: " +
                    self.top_level_log,
                    "blue",
                    attrs=[]))
        else:
            raise Exception(f"tmp_dir ({top_level_log_root}) doesn't exist")

        self.pop = None
        # total number of generated agents from now on (including discarded agents)
        self.num_total_agents = 0

        self.inner_selector = None
        self.mutator = None

        self.evolution_tables_train = []
        self.evolution_tables_test = []

    def _init_pop(self):
        """
        Initialize the population
        """

        if not len(self.resume_from_gen):
            self.pop = [self.agent_factory(i) for i in range(self.pop_sz)]
            self.pop = _mutate_initial_prior_pop(self.pop, self.mutator,
                                                 self.agent_factory)
            self.starting_gen = 0
        else:
            print("resuming from gen :", self.resume_from_gen["gen"])
            self.pop = self.resume_from_gen["init_pop"]
            assert len(self.pop) == self.pop_sz, "wrong initial popluation size"
            for x_i in range(self.pop_sz):
                self.pop[x_i].reset_tracking_attrs()
                self.pop[x_i]._idx = x_i
                self.pop[x_i]._parent_idx = -1
                self.pop[x_i]._root = self.pop[x_i]._idx
                self.pop[x_i]._created_at_gen = -1
            self.starting_gen = self.resume_from_gen["gen"] + 1

        self.num_total_agents = self.pop_sz

    def _get_offspring(self):
        """
        Returns the offspring of the current population
        """

        offsprings = _mutate_prior_pop(self.off_sz, self.pop, self.mutator,
                                       self.agent_factory,
                                       self.num_total_agents)
        self.num_total_agents += len(offsprings)
        
        tmp_pop = self.pop + offsprings  # don't change the order of this concatenation

        return tmp_pop
    
    def _get_metadata(self, pop, type_run):
        """
        Returns the metadata from the tasks : [roots, depths, populations]
        type_run : str "test" or "train"
        """

        if type_run == "train":
            sampler = self.train_sampler
            nb_samples = self.num_train_samples
        else:
            sampler = self.test_sampler
            nb_samples = self.num_test_samples

        metadata = list(
            futures.map(
                ns_instance,
                [
                    sampler
                    for i in range(nb_samples)
                ],  # sampler
                [
                    [x for x in pop]
                    for i in range(nb_samples)
                ],  # population
                [
                    self.mutator
                    for i in range(nb_samples)
                ],  # mutator
                [
                    self.inner_selector
                    for i in range(nb_samples)
                ],  # inner_selector
                [
                    self.agent_factory
                    for i in range(nb_samples)
                ],  # make_ag
                [
                    self.G_inner
                    for i in range(nb_samples)
                ]))  # G_inner

        return metadata
    
    @abstractmethod
    def _meta_learning(self):
        """
        Meta learning function
        """

        return None

    def _update_evolution_table(self, metadata, evolution_table, idx_to_row, samples, tmp_pop=None):
        """
        Updates the action table
        """

        if tmp_pop is not None:
            idx_to_individual = {x._idx: x for x in tmp_pop}

        for pb_i in range(self.num_train_samples):
            rt_i = metadata[pb_i][0]  # roots
            d_i = metadata[pb_i][1]  # depths

            for rt in rt_i:
                if tmp_pop is not None:
                    idx_to_individual[rt]._useful_evolvability += 1
                    idx_to_individual[rt]._adaptation_speed_lst.append(d_i)

                evolution_table[idx_to_row[rt], pb_i] = d_i

        if tmp_pop is not None:
            for ind in tmp_pop:
                if len(ind._adaptation_speed_lst):
                    ind._mean_adaptation_speed = np.mean(
                        ind._adaptation_speed_lst)

    def _save_evolution_table(self, evolution_table, type_save, current_index):
        """
        Saving the evolution table
        type_save : "test" or "train"
        """

        evolution_tables = self.evolution_tables_train if type_save == "train" else self.evolution_tables_test
        evolution_tables.append(evolution_table)

        np.savez_compressed(
            self.top_level_log + "/evolution_table_{}_".format(type_save) +
            str(current_index + self.starting_gen),
            evolution_table)
    
    def __call__(self, disable_testing=False, test_first=False):
        """
        Outer loop of the meta algorithm
        """

        if None in [self.inner_selector, self.mutator]:
            raise ValueError("An attribute wasn't defined properly")

        if self.pop is None:
            self._init_pop()

        for outer_g in range(self.G_outer):
            print("Outer g : {}/{}".format(outer_g, self.G_outer))

            tmp_pop = self._get_offspring()

            evolution_table = -1 * np.ones(
                [len(tmp_pop), self.num_train_samples]
            )  # evolution_table[i,j]=k means that agent i solves env j after k mutations
            idx_to_row = {tmp_pop[i]._idx: i for i in range(len(tmp_pop))}

            if isinstance(self.train_sampler,
                          MetaworldProblems.SampleSingleExampleFromML10):
                ml10obj = metaworld.ML10()

            if not test_first:

                if isinstance(self.train_sampler,
                              MetaworldProblems.SampleSingleExampleFromML10):

                    self.train_sampler.set_ml10obj(ml10obj)

                # [roots, depths, populations]
                metadata = self._get_metadata(tmp_pop, "train")

                self._update_evolution_table(
                    metadata, evolution_table, idx_to_row, self.num_train_samples, tmp_pop)

                # now the meta training part
                self._meta_learning(metadata, tmp_pop)

                with open(
                        self.top_level_log + "/population_prior_" +
                        str(outer_g + self.starting_gen), "wb") as fl:
                    pickle.dump(self.pop, fl)

                self._save_evolution_table(evolution_table, "train", outer_g)

                # reset evolvability and adaptation stats
                for ind in self.pop:
                    ind._useful_evolvability = 0
                    ind._mean_adaptation_speed = float("inf")
                    ind._adaptation_speed_lst = []

            if outer_g % 10 == 0 and not disable_testing:

                test_first = False

                test_evolution_table = -1 * np.ones(
                    [self.pop_sz, self.num_test_samples])
                idx_to_row_test = {
                    self.pop[i]._idx: i
                    for i in range(len(self.pop))
                }

                if isinstance(self.test_sampler,
                              MetaworldProblems.SampleSingleExampleFromML10):
                    self.test_sampler.set_ml10obj(ml10obj)

                test_metadata = self._get_metadata(self.pop, "test")

                self._update_evolution_table(
                    test_metadata, test_evolution_table, idx_to_row_test, self.num_test_samples)
                self._save_evolution_table(
                    test_evolution_table, "test", outer_g)

    def test_population(self, population, in_problem):
        """
        used for a posteriori testing after training is done

        make sure in_problem is passed by reference
        """

        population_size = len(population)
        offsprings_size = population_size

        nov_estimator = NoveltyEstimators.ArchiveBasedNoveltyEstimator(k=15)
        arch = Archives.ListArchive(max_size=5000,
                                    growth_rate=6,
                                    growth_strategy="random",
                                    removal_strategy="random")

        ns = NS.NoveltySearch(
            archive=arch,
            nov_estimator=nov_estimator,
            mutator=self.mutator,
            problem=in_problem,
            selector=self.inner_selector,
            n_pop=population_size,
            n_offspring=offsprings_size,
            agent_factory=self.agent_factory,
            visualise_bds_flag=1,  # log to file
            map_type="scoop",  # or "std"
            logs_root="tmp/test_dir_tmp/",
            compute_parent_child_stats=0,
            initial_pop=[x for x in population])
        # do NS
        nov_estimator.log_dir = ns.log_dir_path
        ns.disable_tqdm = True
        ns.save_archive_to_file = False
        _, solutions = ns(
            iters=self.G_inner,
            stop_on_reaching_task=True,  # do not set to False with current implem
            save_checkpoints=0
        )  # save_checkpoints is not implemented but other functions already do its job

        assert len(
            solutions.keys()
        ) <= 1, "solutions should only contain solutions from a single generation"
        if len(solutions.keys()):
            depth = list(solutions.keys())[0]
        else:
            depth = 100000

        return depth
