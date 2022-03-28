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

import deap
import functools
import numpy as np

from deap import tools as deap_tools

import utils_misc

from meta_learning.class_sparse_rewards import ForSparseRewards


class FAERY(ForSparseRewards):
    """
    FAERY algorithm
    """

    def __init__(self, *args, save_scores=False, **kwargs):
        
        if "name_prefix" not in kwargs.keys():
            kwargs["name_prefix"] = "FAERY"

        super().__init__(*args, **kwargs)

        self.mutator = functools.partial(deap_tools.mutPolynomialBounded,
                                         eta=10,
                                         low=-1.0,
                                         up=1.0,
                                         indpb=0.1)

        self.inner_selector = None # To be set depending on used inner algorithm

        self.save_scores = save_scores
    
    def _get_meta_objectives(self, ind):
        """
        Returns the meta-scores of a given individual
        """

        return [ind._useful_evolvability, -1 * ind._mean_adaptation_speed]
    
    def _meta_learning(self, metadata, tmp_pop):
        """
        Meta learning algorithm for FAERY
        """

        light_pop = []
        for i in range(len(tmp_pop)):
            light_pop.append(deap.creator.LightIndividuals())
            light_pop[-1].fitness.setValues(
                self._get_meta_objectives(tmp_pop[i])
            )
            light_pop[-1].ind_i = i

        chosen_inds = [x.ind_i for x in deap.tools.selNSGA2(light_pop,
                                                            self.pop_sz,
                                                            nd="standard")]
        self.pop = [tmp_pop[u] for u in chosen_inds]

    def _save_meta_objectives(self, tmp_pop, current_index, type_run):
        """
        Saves the meta objectives of the whole population
        """

        np.savez_compressed(
            "{}/meta-scores_{}_{}".format(self.top_level_log, type_run,
                                          str(current_index + self.starting_gen)),
            np.array([self._get_meta_objectives(ind) for ind in tmp_pop])
        )

    def _make_evolution_table(self, metadata, tmp_pop, current_index, type_run="train", save=True):
        super()._make_evolution_table(metadata, tmp_pop, current_index, type_run, save)
        
        if self.save_scores is True and tmp_pop is not None:
            self._save_meta_objectives(tmp_pop, current_index, type_run)


class FAERYQD(FAERY):
    """
    FAERY applied on QD algorithms
    """

    def __init__(self, *args, **kwargs):

        if "name_prefix" not in kwargs.keys():
            kwargs["name_prefix"] = "FAERY_QD"

        super().__init__(*args, **kwargs)

        self.NSGA2 = utils_misc.NSGA2(k=15)
        self.inner_selector = self.NSGA2


class FAERYNS(FAERY):
    """
    FAERY applied on NS algorithms
    """

    def __init__(self, *args, **kwargs):

        if "name_prefix" not in kwargs.keys():
            kwargs["name_prefix"] = "FAERY_NS"

        super().__init__(*args, **kwargs)

        self.inner_selector = functools.partial(utils_misc.selBest,
                                                k=2 * self.pop_sz)


class FAERYRANDOM(FAERY):
    """
    FAERY applied on a RANDOM algorithm that randomly selects the descendants among the parent population
    """

    def __init__(self, *args, **kwargs):

        if "name_prefix" not in kwargs.keys():
            kwargs["name_prefix"] = "FAERY_RAND"

        super().__init__(*args, **kwargs)

        self.inner_selector = functools.partial(deap_tools.selRandom,
                                                k = 2 * self.pop_sz)
                                        

class FAERYRANDOM_COMPLETE(FAERY):
    """
    FAERY applied on a RANDOM algorithm that randomly generates a new population at each step
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name_prefix="FAERY_NS", **kwargs)
        
        self.inner_selector = functools.partial(self.agent_factory)
    
    def _random_pop(self, pop=None):
        return [self.agent_factory(i) for i in range(2 * self.pop_sz)]
