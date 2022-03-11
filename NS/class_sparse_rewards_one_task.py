import random
import argparse
import functools

from deap import tools as deap_tools

import utils_misc as utils_misc

from main import init_main
from class_sparse_rewards import ForSparseRewards


class NSOneTask(ForSparseRewards):
    """
    Running NS on tasks, without any meta learning
    """

    def __init__(self, *args, name_prefix="NS_one", **kwargs):
        super().__init__(*args, name_prefix=name_prefix, **kwargs)

        self.mutator = functools.partial(deap_tools.mutPolynomialBounded,
                                         eta=10,
                                         low=-1.0,
                                         up=1.0,
                                         indpb=0.1)


        self.inner_selector = functools.partial(utils_misc.selBest,
                                                k = 2 * self.pop_sz)
        
        # Running NS a single time
        self.G_outer = 1
    
    def _meta_learning(self, metadata, tmp_pop):
        """
        No meta learning
        """

        return None
    
    def __call__(self, disable_testing=False, test_first=False):
        return super().__call__(disable_testing=True)


class RandomOneTask(ForSparseRewards):
    """
    Running a random pop on tasks, without any meta learning
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, name_prefix="random_one", **kwargs)

        self.mutator = functools.partial(deap_tools.mutPolynomialBounded,
                                         eta=10,
                                         low=-1.0,
                                         up=1.0,
                                         indpb=0.1)

        self.inner_selector = functools.partial(deap_tools.selRandom,
                                                k = 2 * self.pop_sz)

        # Running with random pop a single time
        self.G_outer = 1
    
    def _meta_learning(self, metadata, tmp_pop):
        """
        No meta learning
        """

        return None
    
    def __call__(self, disable_testing=False, test_first=False):
        return super().__call__(disable_testing=True)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='meta experiments')

    parser.add_argument(
        "--problem",
        type=str,
        help="metaworld_ml1, metaworld_ml10, random_mazes",
        default="metaworld_ml1"
    )
    parser.add_argument(
        '--resume',
        type=str,
        help="path to a file population_prior_i with i a generation number",
        default=""
    )
    parser.add_argument(
        '--path_train',
        type=str,
        help="path where *xml and associated *bpm files for training can be found (see ../environments/mazegenerator)",
        default=""
    )
    parser.add_argument(
        '--path_test',
        type=str,
        help="path where *xml and associated *bpm files for testing can be found (see ../environments/mazegenerator)",
        default=""
    )
    parser.add_argument(
        '--maze_size',
        type=int,
        help="size of the maze (8 or 10)",
        default=8
    )
    parser.add_argument(
        '--task_name',
        type=str,
        help="task name for metaworld_ml1 (has to be in metaworld.ML1.ENV_NAMES",
        default="assembly-v2"
    )

    parser.add_argument(
        '--pop_size',
        type=int,
        help="size of the inner loop's population",
        default=40
    )
    parser.add_argument(
        '--off_size',
        type=int,
        help="size of the inner loop's offspring population",
        default=40
    )
    parser.add_argument(
        '--outer_steps',
        type=int,
        help="number of steps for the outer loop",
        default=50
    )
    parser.add_argument(
        '--inner_steps',
        type=int,
        help="number of steps for the inner loop",
        default=200
    )
    parser.add_argument(
        '--nb_samples_train',
        type=int,
        help="number of tasks for the training process",
        default=50
    )
    parser.add_argument(
        '--nb_samples_test',
        type=int,
        help="number of tasks for the testing process",
        default=50
    )
    parser.add_argument(
        '--algo',
        type=str,
        help="type of the algorithm to use",
        default="ns"
    )

    args_obj = parser.parse_args()
    
    train_sampler, test_sampler, agent_factory, top_level_log_root, resume_dict, experiment_config = init_main(args_obj)

    if args_obj.algo.lower() == "ns":
        print("Starting ns")
        ns_one_task = NSOneTask(pop_sz=args_obj.pop_size,
                                off_sz=args_obj.off_size,
                                G_outer=args_obj.outer_steps,
                                G_inner=args_obj.inner_steps,
                                train_sampler=train_sampler,
                                test_sampler=test_sampler,
                                num_train_samples=args_obj.nb_samples_train,
                                num_test_samples=args_obj.nb_samples_test,
                                agent_factory=agent_factory,
                                top_level_log_root=top_level_log_root,
                                resume_from_gen=resume_dict)
        
        ns_one_task()

    else:
        print("Starting random")
        random_one_task = RandomOneTask(pop_sz=args_obj.pop_size,
                                    off_sz=args_obj.off_size,
                                    G_outer=args_obj.outer_steps,
                                    G_inner=args_obj.inner_steps,
                                    train_sampler=train_sampler,
                                    test_sampler=test_sampler,
                                    num_train_samples=args_obj.nb_samples_train,
                                    num_test_samples=args_obj.nb_samples_test,
                                    agent_factory=agent_factory,
                                    top_level_log_root=top_level_log_root,
                                    resume_from_gen=resume_dict)
        
        random_one_task()