import json
import numpy as np

from utils_main import init_main, get_parser
from meta_learning.class_sparse_rewards_faery import FAERY, FAERYQD, FAERYNS, FAERYRANDOM


class FAERY_Ablation(FAERY):
    """
    FAERY, ablation study object
    """

    def __init__(self, *args, objective_to_ignore=0, **kwargs):
        super().__init__(*args, **kwargs)
    
        self.objective_to_ignore = objective_to_ignore
        assert self.objective_to_ignore in range(-1,2), "Wrong objective (-1, 0, 1)"

    def _get_meta_objectives(self, ind):
        score = FAERY._get_meta_objectives(self, ind)

        if self.objective_to_ignore != -1:
            score[self.objective_to_ignore] = 0
                
        return score
    
    def _save_meta_objectives(self, tmp_pop, current_index, type_run):
        """
        Saves the meta objectives of the whole population
        """

        np.savez_compressed(
            "{}/meta-scores_{}_{}".format(self.top_level_log, type_run,
                                          str(current_index + self.starting_gen)),
            np.array([FAERY._get_meta_objectives(self, ind) for ind in tmp_pop])
        )


class FAERY_Ablation_QD(FAERYQD, FAERY_Ablation):

    def __init__(self, *args, **kwargs):

        if "name_prefix" not in kwargs.keys():
            kwargs["name_prefix"] = "FAERY_QD_{}".format(kwargs["objective_to_ignore"])

        super().__init__(*args, **kwargs)


class FAERY_Ablation_NS(FAERYNS, FAERY_Ablation):

    def __init__(self, *args, **kwargs):

        if "name_prefix" not in kwargs.keys():
            kwargs["name_prefix"] = "FAERY_NS_{}".format(kwargs["objective_to_ignore"])

        super().__init__(*args, **kwargs)


class FAERY_Ablation_RANDOM(FAERYRANDOM, FAERY_Ablation):

    def __init__(self, *args, **kwargs):

        if "name_prefix" not in kwargs.keys():
            kwargs["name_prefix"] = "FAERY_RANDOM_{}".format(kwargs["objective_to_ignore"])

        super().__init__(*args, **kwargs)
        

if __name__=="__main__":

    parser = get_parser()

    parser.add_argument(
        "--to_remove",
        type=int,
        help="which objective to remove : 0, 1 or -1 if none",
        default=-1
    )

    args_obj = parser.parse_args()
    assert args_obj.to_remove in [-1, 0, 1], "Can't remove given objective"

    train_sampler, test_sampler, agent_factory, top_level_log_root, resume_dict, experiment_config = init_main(args_obj)

    #  STARTING UP THE ALGORITHM
    if args_obj.inner_algorithm.lower() == "qd":
        algo_obj = FAERY_Ablation_QD
    elif args_obj.inner_algorithm.lower() == "ns":
        algo_obj = FAERY_Ablation_NS
    elif args_obj.inner_algorithm.lower() == "random":
        algo_obj = FAERY_Ablation_RANDOM
    else:
        raise TypeError("Unsupported algorithm")

    algo = algo_obj(pop_sz=args_obj.pop_size,
                        off_sz=args_obj.off_size,
                        G_outer=args_obj.outer_steps,
                        G_inner=args_obj.inner_steps,
                        train_sampler=train_sampler,
                        test_sampler=test_sampler,
                        num_train_samples=args_obj.nb_samples_train,
                        num_test_samples=args_obj.nb_samples_test,
                        agent_factory=agent_factory,
                        top_level_log_root=top_level_log_root,
                        resume_from_gen=resume_dict,
                        objective_to_ignore=args_obj.to_remove,
                        save_scores=True)
                    
    with open(algo.top_level_log + "/experiment_config", "w") as fl:
        json.dump(experiment_config, fl)

    algo()
