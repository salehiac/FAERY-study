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

from meta_learning.class_sparse_rewards_faery import FAERY


class FAERY_AUG(FAERY):
    """
    FAERY_AUG algorithm
    """

    def __init__(self, *args, save_scores=True, **kwargs):
        
        if "name_prefix" not in kwargs.keys():
            kwargs["name_prefix"] = "FAERY_AUG"

        super().__init__(*args, **kwargs)

        self.light_ind_type = deap.creator.LightIndividuals3D
    
    def _get_meta_objectives(self, ind):
        """
        Returns the meta-scores of a given individual
        """

        return [
            ind.get_nb_solved_tasks(),
            ind.get_median_solutions_per_tasks(),
            -ind.get_mean_adaptation_speed()
        ]


class FAERY_AUG_QD(FAERY_AUG):
    """
    FAERY_AUG applied on QD algorithms
    """

    def __init__(self, *args, **kwargs):

        if "name_prefix" not in kwargs.keys():
            kwargs["name_prefix"] = "FAERY_AUG_QD"

        super().__init__(*args, **kwargs)

        self.NSGA2 = utils_misc.NSGA2(k=15)
        self.inner_selector = self.NSGA2


class FAERY_AUG_NS(FAERY_AUG):
    """
    FAERY_AUG applied on NS algorithms
    """

    def __init__(self, *args, **kwargs):

        if "name_prefix" not in kwargs.keys():
            kwargs["name_prefix"] = "FAERY_AUG_NS"

        super().__init__(*args, **kwargs)

        self.inner_selector = functools.partial(utils_misc.selBest,
                                                k=2 * self.pop_sz)


class FAERY_AUG_RANDOM(FAERY_AUG):
    """
    FAERY_AUG applied on a RANDOM algorithm that randomly selects the descendants among the parent population
    """

    def __init__(self, *args, **kwargs):

        if "name_prefix" not in kwargs.keys():
            kwargs["name_prefix"] = "FAERY_AUG_RAND"

        super().__init__(*args, **kwargs)

        self.inner_selector = functools.partial(deap_tools.selRandom,
                                                k = 2 * self.pop_sz)
                                        

class FAERY_AUG_RANDOM_COMPLETE(FAERY_AUG):
    """
    FAERY_AUG applied on a RANDOM algorithm that randomly generates a new population at each step
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name_prefix="FAERY_AUG_NS", **kwargs)
        
        self.inner_selector = functools.partial(self.agent_factory)
    
    def _random_pop(self, pop=None):
        return [self.agent_factory(i) for i in range(2 * self.pop_sz)]
