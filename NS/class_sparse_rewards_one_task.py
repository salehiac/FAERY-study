import argparse
import functools

from deap import tools as deap_tools

import utils_misc as utils_misc

from utils_main import init_main, get_parser
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

    parser = get_parser()
    
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