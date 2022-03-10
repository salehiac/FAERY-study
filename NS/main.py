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
import json
import argparse
import pickle
import metaworld

import Agents
import HardMaze
import MetaworldProblems

from population_priors import MetaQDForSparseRewards, NSForSparseRewards


def _make_2d_maze_ag(ag_idx):
    """
    because scoop only likes top-level functions/objects...
    """
    agt = Agents.SmallFC_FW(ag_idx,
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
    agt = Agents.SmallFC_FW(ag_idx,
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
    if args_obj.problem == "random_mazes":

        agent_factory = _make_2d_maze_ag
        top_level_log_root = "tmp/NS_LOGS"

        train_sampler = functools.partial(
            HardMaze.sample_mazes,
            G=args_obj.maze_size,
            xml_template_path="../environments/env_assets/maze_template.xml",
            tmp_dir="tmp/",
            from_dataset=args_obj.path_train,
            random_goals=False)

        test_sampler = functools.partial(
            HardMaze.sample_mazes,
            G=args_obj.maze_size,
            xml_template_path="../environments/env_assets/maze_template.xml",
            tmp_dir="tmp/",
            from_dataset=args_obj.path_test,
            random_goals=False)

        experiment_config["task_name"] = "maze{}x{}".format(
            args_obj.maze_size, args_obj.maze_size)

    elif args_obj.problem == "metaworld_ml1":

        behavior_descr_type = "type_3"
        agent_factory = _make_metaworld_ml1_ag
        top_level_log_root = "tmp/META_LOGS_ML1/"

        train_sampler = MetaworldProblems.SampleFromML1(
            bd_type=behavior_descr_type, mode="train", task_name=args_obj.task_name)
        test_sampler = MetaworldProblems.SampleFromML1(
            bd_type=behavior_descr_type, mode="test", task_name=args_obj.task_name)

        if args_obj.task_name not in metaworld.ML1.ENV_NAMES:
            raise ValueError("--task_name not available for ML1")

        experiment_config["task_name"] = args_obj.task_name

    elif args_obj.problem == "metaworld_ml10":

        behavior_descr_type = "type_3"
        agent_factory = _make_metaworld_ml1_ag
        top_level_log_root = "tmp/META_LOGS_ML10/"

        experiment_config["ML10 called every outer loop"] = 1

        train_sampler = MetaworldProblems.SampleSingleExampleFromML10(
            bd_type=behavior_descr_type, mode="train")
        test_sampler = MetaworldProblems.SampleSingleExampleFromML10(
            bd_type=behavior_descr_type, mode="test")
    
    return train_sampler, test_sampler, agent_factory, top_level_log_root, resume_dict, experiment_config


if __name__ == "__main__":

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

    args_obj = parser.parse_args()
    
    train_sampler, test_sampler, agent_factory, top_level_log_root, resume_dict, experiment_config = init_main(args_obj)

    #  STARTING UP THE ALGORITHM
    algo = MetaQDForSparseRewards(pop_sz=args_obj.pop_size,
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

    # Test algorithm
    # algo = NSForSparseRewards(pop_sz=args_obj.pop_size,
    #                           off_sz=args_obj.off_size,
    #                           G_outer=args_obj.outer_steps,
    #                           G_inner=args_obj.inner_steps,
    #                           train_sampler=train_sampler,
    #                           test_sampler=test_sampler,
    #                           num_train_samples=args_obj.nb_samples_train,
    #                           num_test_samples=args_obj.nb_samples_test,
    #                           agent_factory=agent_factory,
    #                           top_level_log_root=top_level_log_root,
    #                           resume_from_gen=resume_dict)

    with open(algo.top_level_log + "/experiment_config", "w") as fl:
        json.dump(experiment_config, fl)

    algo()
