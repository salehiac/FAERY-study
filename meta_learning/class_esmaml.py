import os

import numpy as np

from abc import ABC
from termcolor import colored
from scoop import futures

import utils_misc

from meta_learning.utils_maml import get_reward


# NEED TO ADD TESTING GRADIENT STEPS?
class ESMAML(ABC):
    """
    Implementation of Zero-Order ES-MAML with ES Gradient Adaptation for Meta-world
    """

    def __init__(self,

                 agent_factory,

                 train_sampler,
                 test_sampler,

                 num_train_samples=50,
                 num_test_samples=5,
                 test_freq=1,

                 G_outer=150,

                 alpha=5e-2,
                 beta=1e-2,
                 K=5,
                 sigma=1e-1,
                 
                 theta = None,

                 top_level_log_root="tmp/",
                 name_prefix="es-maml"):
        """
        agent_factory : initialize a policy (problem dependant)

        train_sampler, test_sampler : tasks generators for training and testing (have to be callable)
        /!\ samplers currently work with replacement

        num_train_samples, num_test_samples : number of tasks to sample at each step (can influence algorithm performances)
        test_freq : frequency of testing runs

        G_outer : Number of steps to run ES-MAML for

        alpha : adaptation step_size (weighting random vectors)
        beta : meta step size (weighting theta adjustments)
        K : number of queries for each task (inside ES Grad)
        sigma : precision of gradient
        
        theta: initial policy (None will generate it with agent_factory)

        top_level_log_root, name_prefix : saving parameters
        """
        
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.sigma = sigma
        self.theta = theta

        self.G_outer = G_outer

        self.train_sampler = train_sampler
        self.test_sampler = test_sampler

        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples

        self.agent_factory = agent_factory
        
        self.test_freq = test_freq
        
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

    def _init_theta(self):
        """
        Initialize the policy
        """

        self.theta = self.agent_factory(0)
        self.starting_gen = 0
    
    def test_policy(self, step, save=True):
        """
        Tests the policy
        """

        random_tasks = [self.test_sampler(num_samples=1) for _ in range(self.num_test_samples)]

        all_results = list(
            futures.map(
                random_tasks,
                [self.theta] * self.num_test_samples
            )
        )

        to_save = [
            (result[0], result[2]) #fitness, is_task_solved
            for result in all_results
        ]
        
        if save is True:
            utils_misc.dump_pickle(
                "{}/test_results_{}".format(self.top_level_log, str(step + self.starting_gen)),
                to_save
            )

    def __call__(self, disable_testing=False, test_first=False, save=True):
        """
        Outer loop of the meta algorithm
        """

        if self.theta is None:
            self._init_theta()

        if test_first is True:
            self.test_policy(step=0, save=save)

        for outer_g in range(self.G_outer):
            print("Outer g : {}/{}".format(outer_g, self.G_outer))

            size = len(self.theta.get_flattened_weights())
            random_tasks = [self.train_sampler(num_samples=1) for _ in range(self.num_train_samples)]
            random_vectors = [np.random.normal(size=size) for _ in range(self.num_train_samples)]

            all_values = list(
                futures.map(
                    get_reward,
                    [self] * self.num_train_samples,
                    random_tasks,
                    random_vectors
                )
            )

            self.theta = self.theta + self.beta / self.sigma * sum([all_values[i] * random_vectors[i] for i in range(self.num_train_samples)]) / self.num_train_samples

            if save is True:
                utils_misc.dump_pickle(
                    "{}/policy_{}".format(self.top_level_log, str(outer_g + self.starting_gen)),
                    self.theta
                )

            if outer_g % self.test_freq == 0 and not disable_testing:
                self.test_policy(step=outer_g, save=save)
