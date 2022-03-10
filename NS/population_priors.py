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

import functools
import random
import deap

import numpy as np

from deap import tools as deap_tools

import MiscUtils

from ForSparseRewards import ForSparseRewards


class MetaQDForSparseRewards(ForSparseRewards):
    """
    FAERY algorithm
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name_prefix="FAERY", **kwargs)

        self.mutator = functools.partial(deap_tools.mutPolynomialBounded,
                                         eta=10,
                                         low=-1.0,
                                         up=1.0,
                                         indpb=0.1)

        self.NSGA2 = MiscUtils.NSGA2(k=15)
        self.inner_selector = self.NSGA2

    def _meta_learning(self, metadata, tmp_pop):
        """
        Meta learning algorithm for FAERY
        """

        light_pop = []
        for i in range(len(tmp_pop)):
            light_pop.append(deap.creator.LightIndividuals())
            light_pop[-1].fitness.setValues(
                [
                    tmp_pop[i]._useful_evolvability,
                    -1 * (tmp_pop[i]._mean_adaptation_speed)
                ]
            )  # the -1 factor is because we want to minimise that speed
            light_pop[-1].ind_i = i

        chosen_inds = [x.ind_i for x in deap.tools.selNSGA2(light_pop,
                                                            self.pop_sz,
                                                            nd="standard")]
        self.pop = [tmp_pop[u] for u in chosen_inds]


class NSForSparseRewards(ForSparseRewards):
    """
    Simple NS algorithm applied to Multi-Tasks
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name_prefix="NS", **kwargs)

        self.mutator = functools.partial(deap_tools.mutPolynomialBounded,
                                         eta=10,
                                         low=-1.0,
                                         up=1.0,
                                         indpb=0.1)

        self.inner_selector = functools.partial(MiscUtils.selBest,
                                                k=2 * self.pop_sz)

    def _meta_learning(self, metadata, tmp_pop):
        """
        Meta learning algorithm for simple NS
        """

        sols_lst = list(set(np.concatenate([m[2] for m in metadata])))
        self.pop = random.choices(sols_lst, k=self.pop_sz)
