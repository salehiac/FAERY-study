import itertools
import os
import pickle
import numpy as np

from abc import ABC, abstractmethod
from termcolor import colored
from scoop import futures
from itertools import product

import utils_misc
from meta_learning.utils_maml import es_grad


class ESMAML(ABC):
    """
    ESMAML implementation for Meta-world
    """

    def __init__(self,
                
                 G_outer,
                 alpha, beta, K, sigma,

                 train_sampler,

                 num_train_samples,

                 agent_factory,

                 top_level_log_root="tmp/",
                 name_prefix="es-maml"):
        
        self.alpha, self.beta, self.K, self.sigma, self.theta = alpha, beta, K, sigma, None

        self.G_outer = G_outer

        self.train_sampler = train_sampler

        self.num_train_samples = num_train_samples

        self.agent_factory = agent_factory
        
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

        self.evolution_tables_train = []
        self.evolution_tables_test = []

    def _init_theta(self):
        """
        Initialize the policy
        """

        self.theta = self.agent_factory(0)
        self.starting_gen = 0
    
    def _get_reward_esgrad(self, task, random_vector):
        """
        Returns [...]
        """        

        d = es_grad(task, self.theta + self.sigma * random_vector, self.K, self.sigma)
        return task(self.theta + self.sigma * random_vector + self.alpha * d)[0]
        # STILL NEED TO GATHER METADATA FOR PLOTTING PURPOSES

    def __call__(self, save=True):
        """
        Outer loop of the meta algorithm
        """

        if self.theta is None:
            self._init_theta()

        for outer_g in range(self.G_outer):
            print("Outer g : {}/{}".format(outer_g, self.G_outer))
                
            random_tasks = [self.train_sampler(num_samples=1) for _ in range(self.num_train_samples)]
            random_vectors = [np.random.normal(size=self.theta.shape) for _ in range(self.n)]

            all_values = list(
                futures.map(
                    self._get_metadata,
                    random_tasks,
                    random_vectors
                )
            )

            self.theta += self.beta / self.sigma * sum([all_values[i] * random_tasks[i] for i in range(self.num_train_samples)]) / self.num_train_samples

            if save is True:
                utils_misc.dump_pickle(
                    "{}/policy_{}".format(self.top_level_log, str(outer_g + self.starting_gen)),
                    self.theta
                )                