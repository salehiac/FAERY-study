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

import json

from utils_main import init_main, get_parser
from class_sparse_rewards_faery import FAERYNS, FAERYQD, FAERYRANDOM, FAERYRANDOM_COMPLETE


if __name__ == "__main__":

    args_obj = get_parser().parse_args()
    train_sampler, test_sampler, agent_factory, top_level_log_root, resume_dict, experiment_config = init_main(args_obj)

    #  STARTING UP THE ALGORITHM
    if args_obj.inner_algorithm.lower() == "qd":
        algo_obj = FAERYQD
    elif args_obj.inner_algorithm.lower() == "ns":
        algo_obj = FAERYNS
    elif args_obj.inner_algorithm.lower() == "random":
        algo_obj = FAERYRANDOM
    else:
        algo_obj = FAERYRANDOM_COMPLETE

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
                    resume_from_gen=resume_dict)
                    
                    
    with open(algo.top_level_log + "/experiment_config", "w") as fl:
        json.dump(experiment_config, fl)

    algo()
