import functools
import random
import numpy as np
import deap.tools as deap_tools

import utils_misc
import class_problem_metaworld

from class_sparse_rewards import ForSparseRewards
from utils_sparse_rewards import ns_instance


class MetaNS(ForSparseRewards):
    """
    Simple NS algorithm applied to Multi-Tasks
    """

    def __init__(self, *args, **kwargs):

        if "name_prefix" not in kwargs.keys():
            kwargs["name_prefix"] = "NS"

        super().__init__(*args, **kwargs)

        self.mutator = functools.partial(deap_tools.mutPolynomialBounded,
                                         eta=10,
                                         low=-1.0,
                                         up=1.0,
                                         indpb=0.1)

        self.inner_selector = functools.partial(utils_misc.selBest,
                                                k=2 * self.pop_sz)

    def _meta_learning(self, metadata, tmp_pop):
        """
        Meta learning algorithm for simple NS
        """

        sols_lst = list(set(np.concatenate([m[2] for m in metadata])))
        self.pop = random.choices(sols_lst, k=self.pop_sz)


class MultiTaskNS(MetaNS):
    """
    NS algorithm applied to all the tasks at once
    """

    def __init__(self, *args, nb_tasks=10, task_name="assembly-v2", behavior_descr_type = "type_3", **kwargs):
        super().__init__(*args, name_prefix="NS_MT", **kwargs)

        self.nb_tasks = nb_tasks

        base_args = {"bd_type":behavior_descr_type, "task_name":task_name, "nb_tasks":self.nb_tasks}
        self.train_sampler = class_problem_metaworld.SampleFromML1(mode="train", **base_args)
        self.test_sampler = class_problem_metaworld.SampleFromML1(mode="test", **base_args)

    def _get_metadata(self, pop, type_run):

        if type_run == "train":
            sampler = self.train_sampler
            nb_samples = self.num_train_samples
        else:
            sampler = self.test_sampler
            nb_samples = self.num_test_samples

        metadata = ns_instance(
            sampler,  # sampler
            [x for x in pop],  # population
            self.mutator,  # mutator
            self.inner_selector,  # inner_selector
            self.agent_factory,  # make_ag
            self.G_inner    # G_inner
        )
        
        return metadata
    
    def _make_evolution_table(self, metadata, tmp_pop, current_index, type_run="train", save=True):
        return super()._make_evolution_table(metadata, tmp_pop, current_index, type_run, save)