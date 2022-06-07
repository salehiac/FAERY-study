import json
import pickle
import argparse
import functools
import metaworld

import problem.class_agent as class_agent
import problem.class_problem_metaworld as class_problem_metaworld


def get_parser():

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
        default=25
    )
    parser.add_argument(
        '--nb_samples_test',
        type=int,
        help="number of tasks for the testing process",
        default=25
    )
    parser.add_argument(
        "--inner_algorithm",
        type=str,
        help="the inner algorithm to use (QD, NS, RANDOM)",
        default="NS"
    )
    parser.add_argument(
        "--top_level_log",
        type=str,
        help="the directory where the logs will be stored",
        default="tmp/"
    )
    parser.add_argument(
        "--steps_after_solved",
        type=int,
        help="maximum number of steps after the first solution is found in each inner algorithm",
        default=0
    )
    parser.add_argument(
        "--test_freq",
        type=int,
        help="frequency of testing",
        default=5
    )

    return parser


def _make_2d_maze_ag(ag_idx):
    """
    because scoop only likes top-level functions/objects...
    """
    agt = class_agent.SmallFC_FW(ag_idx,
                            in_d=5,
                            out_d=2,
                            num_hidden=3,
                            hidden_dim=10,
                            output_normalisation="")
    return agt


def _make_metaworld_ml1_ag(ag_idx):
    """
    because scoop only likes top-level functions/objects...
    """
    agt = class_agent.SmallFC_FW(ag_idx,
                            in_d=39,
                            out_d=4,
                            num_hidden=1,
                            hidden_dim=50,
                            output_normalisation="tanh")
    return agt


def init_main(args_obj):
    #  RESUMING
    resume_dict = {}
    if len(args_obj.resume):
        print("resuming...")
        pop_fn = args_obj.resume
        with open(pop_fn, "rb") as fl:
            resume_dict["init_pop"] = pickle.load(fl)
        dig = [
            x for x in pop_fn[pop_fn.find("population_prior"):]
            if x.isdigit()
        ]
        dig = int(functools.reduce(lambda x, y: x + y, dig, ""))
        resume_dict["gen"] = dig
        print("loaded_init_pop...")

        if args_obj.problem != "metaworld_ml10":
            orig_cfg = functools.reduce(lambda x, y: x + "/" + y,
                                        pop_fn.split("/")[:-1],
                                        "") + "/experiment_config"
            with open(orig_cfg, "r") as fl:
                orig_tsk_name = json.load(fl)["task_name"]
            resuming_from_str = orig_tsk_name + "_" + str(dig)
    else:
        resuming_from_str = ""

    #  SETTING UP EXPERIMENTS
    experiment_config = {
        "pop_sz": args_obj.pop_size,
        "off_sz": args_obj.off_size,
        "num_train_samples": args_obj.nb_samples_train,
        "num_test_samples": args_obj.nb_samples_test,
        "g_outer": args_obj.outer_steps,
        "g_inner": args_obj.inner_steps
    }

    # DIFFERENT SETTINGS FOR EACH ENV
    if args_obj.problem == "metaworld_ml1":

        behavior_descr_type = "type_3"
        agent_factory = _make_metaworld_ml1_ag

        train_sampler = class_problem_metaworld.SampleFromML1(
            bd_type=behavior_descr_type, mode="train", task_name=args_obj.task_name)
        test_sampler = class_problem_metaworld.SampleFromML1(
            bd_type=behavior_descr_type, mode="test", task_name=args_obj.task_name)

        if args_obj.task_name not in metaworld.ML1.ENV_NAMES:
            raise ValueError("--task_name not available for ML1")

        experiment_config["task_name"] = args_obj.task_name

    elif args_obj.problem == "metaworld_ml10":

        behavior_descr_type = "type_3"
        agent_factory = _make_metaworld_ml1_ag

        experiment_config["ML10 called every outer loop"] = 1

        train_sampler = class_problem_metaworld.SampleSingleExampleFromML10(
            bd_type=behavior_descr_type, mode="train")
        test_sampler = class_problem_metaworld.SampleSingleExampleFromML10(
            bd_type=behavior_descr_type, mode="test")

    return train_sampler, test_sampler, agent_factory, args_obj.top_level_log, resume_dict, experiment_config
