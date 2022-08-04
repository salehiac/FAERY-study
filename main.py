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
from meta_learning.class_sparse_rewards_faery import FAERYNS, FAERYQD, FAERYRANDOM, FAERYRANDOM_COMPLETE
from meta_learning.class_sparse_rewards_faery_augmented import FAERY_AUG_NS, FAERY_AUG_QD, FAERY_AUG_RANDOM, FAERY_AUG_RANDOM_COMPLETE
from meta_learning.class_esmaml import ESMAML


# MAIN FILE USED TO RUN FAERY WITH GIVEN PARAMETERS

if __name__ == "__main__":

    args_obj = get_parser().parse_args()
    train_sampler, test_sampler, agent_factory, top_level_log_root, resume_dict, experiment_config = init_main(args_obj)

    #  STARTING UP THE ALGORITHM
    algo_req = args_obj.inner_algorithm.lower()
    
    

    if "maml" in algo_req:
        algo_obj = ESMAML

        algo = algo_obj(
            agent_factory=agent_factory,

            train_sampler=train_sampler,
            test_sampler=test_sampler,

            num_train_samples=args_obj.nb_samples_train,
            num_test_samples=args_obj.nb_samples_test,
            test_freq=args_obj.test_freq,

            G_outer=args_obj.outer_steps,

            alpha=args_obj.alpha,
            beta=args_obj.beta,
            K=args_obj.K,
            sigma=args_obj.sigma,
            
            top_level_log_root=top_level_log_root)
    else:
        list_algo = [FAERYNS, FAERYQD, FAERYRANDOM, FAERYRANDOM_COMPLETE] if "aug" not in algo_req \
            else [FAERY_AUG_NS, FAERY_AUG_QD, FAERY_AUG_RANDOM, FAERY_AUG_RANDOM_COMPLETE]

        if "ns" in algo_req:    algo_obj = list_algo[0]
        elif "qd" in algo_req:  algo_obj = list_algo[1]
        elif "random" in algo_req:  algo_obj = list_algo[2]
        else:   algo_obj = list_algo[3]

        algo = algo_obj(pop_sz=args_obj.pop_size,
                        off_sz=args_obj.off_size,
                        G_outer=args_obj.outer_steps,
                        G_inner=args_obj.inner_steps,
                        train_sampler=train_sampler,
                        test_sampler=test_sampler,
                        num_train_samples=args_obj.nb_samples_train,
                        num_test_samples=args_obj.nb_samples_test,
                        agent_factory=agent_factory,
                        test_freq=args_obj.test_freq,
                        steps_after_solved=args_obj.steps_after_solved,
                        top_level_log_root=top_level_log_root,
                        resume_from_gen=resume_dict,
                        ablation=args_obj.ablation)
                    
                    
    with open(algo.top_level_log + "/experiment_config", "w") as fl:
        json.dump(experiment_config, fl)

    algo()
