import json
import numpy as np

from utils_main import init_main, get_parser
from class_sparse_rewards_faery import FAERYQD


class FAERYQD_Ablation(FAERYQD):
    """
    FAERY applied on QD, ablation study object
    """

    def __init__(self, *args, objective_to_ignore=0, **kwargs):
        super().__init__(*args, name_prefix="FAERY_QD_{}".format(objective_to_ignore), **kwargs)
    
        self.objective_to_ignore = objective_to_ignore

    def _get_meta_objectives(self, ind):
        score = super()._get_meta_objectives(ind)
        score[self.objective_to_ignore] = 0
        return score
    
    def _save_meta_objectives(self, tmp_pop, current_index, type_run):
        """
        Saves the meta objectives of the whole population
        """

        np.savez_compressed(
            "{}/meta-scores_{}_{}".format(self.top_level_log, type_run,
                                          str(current_index + self.starting_gen)),
            np.array([FAERYQD._get_meta_objectives(self, ind) for ind in tmp_pop])
        )


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
    if args_obj.to_remove == -1:
        algo_obj = FAERYQD
    else:
        algo_obj = FAERYQD_Ablation

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
                    save_scores=True)
    
    if args_obj.to_remove != -1:
        algo_obj.objective_to_ignore = args_obj.to_remove
                    
    with open(algo.top_level_log + "/experiment_config", "w") as fl:
        json.dump(experiment_config, fl)

    algo()
