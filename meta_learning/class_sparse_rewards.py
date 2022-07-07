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

import os
import pickle
import numpy as np

from abc import ABC, abstractmethod
from termcolor import colored
from scoop import futures

import utils_misc

from meta_learning.utils_sparse_rewards import _mutate_initial_prior_pop, _mutate_prior_pop, ns_instance


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
                 test_freq=10,
                 steps_after_solved=0,
                 top_level_log_root="tmp/",
                 name_prefix="meta-learning",
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
        self.test_freq = test_freq
        self.steps_after_solved = steps_after_solved
        self.resume_from_gen = resume_from_gen
        self.name_prefix = name_prefix
        self.folder_name = name_prefix + '_' + utils_misc.rand_string()

        if os.path.isdir(top_level_log_root):
            self.top_level_log = utils_misc.create_directory(
                dir_basename= "{}/{}".format(top_level_log_root, self.folder_name),
                remove_if_exists=True,
                pid=True)
            
            utils_misc.create_directory(
                dir_basename= "{}/{}".format(self.top_level_log, "NS_LOGS") ,
                remove_if_exists=True,
                pid=False)

            print(
                colored(
                    "[NS info] temporary dir for {} was created: ".format(name_prefix) +
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
                self.pop[x_i].reset()
                self.pop[x_i]._idx = x_i
                self.pop[x_i]._parent_idx = -1
                self.pop[x_i]._root = self.pop[x_i]._idx
                self.pop[x_i]._created_at_gen = -1
                self.pop[x_i]._distance_to_root = 0
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
    
    def _get_metadata(self, pop, type_run, outer_g):
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

        # FOR DEBUGING PURPOSES
        # metadata = [
        #     ns_instance(
        #         sampler,  # sampler
        #         [x for x in pop],  # population
        #         self.mutator,  # mutator
        #         self.inner_selector,  # inner_selector
        #         self.agent_factory,  # make_ag
        #         self.G_inner,    # G_inner
        #         self.top_level_log,
        #         (type_run, outer_g),
        #         self.steps_after_solved,
        #         k,
        #     ) for k in range(nb_samples)
        # ]
        
        metadata = list(
            futures.map(
                ns_instance,
                [sampler] * nb_samples,  # sampler
                [[x for x in pop] for _ in range(nb_samples)],  # population
                [self.mutator] * nb_samples,  # mutator
                [self.inner_selector] * nb_samples,  # inner_selector
                [self.agent_factory] * nb_samples,  # make_ag
                [self.G_inner] * nb_samples,    # G_inner
                [self.top_level_log] * nb_samples,
                [(type_run, outer_g)] * nb_samples,
                [self.steps_after_solved] * nb_samples,
                list(range(nb_samples))
            )
        )

        return metadata
    
    @abstractmethod
    def _meta_learning(self):
        """
        Meta learning function
        """

        return None

    def _update_evolution_table(self, metadata, evolution_table, idx_to_row, tmp_pop=None):
        """
        Updates the evolution table
        """

        assert tmp_pop is not None, "tmp_pop should not be None" # Removed if tmp_pop is not None...

        idx_to_individual = {x._idx: x for x in tmp_pop}
        
        for x in tmp_pop:
            x.nb_solutions = 0
            x.instance_to_nb_solutions = [0] * len(metadata)
            x.instance_to_adaptation_speeds = [[] for _ in range(len(metadata))]
            x.instance_to_mutation_distance = [[] for _ in range(len(metadata))]
            
        instances_to_roots = {set() for _ in range(len(metadata))}
        for instance, (parents, solutions) in enumerate(metadata):
            for iteration, list_solutions in solutions.items():
                for solver in list_solutions:
                    root = solver._root
                    instances_to_roots[instance].add(root)

                    root_ind = idx_to_individual[root]

                    root_ind.nb_solutions += 1
                    root_ind.instance_to_nb_solutions[instance] += 1
                    root_ind.instance_to_adaptation_speeds[instance].append(iteration)
                    root_ind.instance_to_mutation_distance[instance].append(solver._distance_to_root)  
        
        for instance, all_roots in instances_to_roots.items():
            for root in all_roots:
                root_ind = idx_to_individual[root]
                
                evolution_table[idx_to_row[root], instance, 0] = np.min(
                    root_ind.instance_to_mutation_distance[instance]
                )
                evolution_table[idx_to_row[root], instance, 1] = np.min(
                    root_ind.instance_to_adaptation_speeds[instance]
                )

    def _save_evolution_table(self, evolution_table, type_save, current_index):
        """
        Saving the evolution table
        type_save : "test" or "train"
        """

        np.savez_compressed(
                "{}/evolution_table_{}_{}".format(
                    self.top_level_log, type_save,
                    str(current_index + self.starting_gen)
                ),
                evolution_table
        )
    
    def _make_evolution_table(self, metadata, tmp_pop, current_index, type_run="train", save=True):
        """
        Make and saves the evolution table from the given metadata
        """

        if type_run == "train":
            # evolution_table[i,j]=k means that agent i solves env j after (k mutations, l steps)
            evolution_table = -1 * np.ones([len(tmp_pop), self.num_train_samples, 2])  
            idx_to_row = {tmp_pop[i]._idx:i for i in range(len(tmp_pop))}
        elif type_run == "test":
            evolution_table = -1 * np.ones([self.pop_sz, self.num_test_samples, 2])
            idx_to_row = {self.pop[i]._idx:i for i in range(len(self.pop))}
        else:
            raise ValueError("Unknown type of run")
        
        self._update_evolution_table(metadata, evolution_table, idx_to_row, tmp_pop)

        if save is True:
            self._save_evolution_table(evolution_table, type_run, current_index)

    def __call__(self, disable_testing=False, test_first=False, save=True):
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

            if not test_first:

                # [parents, solutions per step]
                metadata = self._get_metadata(tmp_pop, "train", outer_g)

                # Updating the evolution table and the individuals
                self._make_evolution_table(metadata, tmp_pop, outer_g, 'train', save)

                # now the meta training part
                self._meta_learning(metadata, tmp_pop)

                if save is True:
                    utils_misc.dump_pickle(
                        "{}/population_prior_{}".format(self.top_level_log, str(outer_g + self.starting_gen)),
                        self.pop
                    )                

            if outer_g % self.test_freq == 0 and not disable_testing:

                test_first = False
                metadata = self._get_metadata(self.pop, "test", outer_g)
                
                self._make_evolution_table(metadata, tmp_pop, outer_g, "test", save)
