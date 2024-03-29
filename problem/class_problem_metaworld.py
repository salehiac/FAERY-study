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

import time
import random
import gc
import torch
import metaworld
import numpy as np
import matplotlib.pyplot as plt

import problem.class_behavior_descr as class_behavior_descr

from problem.class_problem import Problem
from problem.class_env_multitask import MultiTaskEnv


class SampleFromML1:

    def __init__(self, bd_type, mode, task_name):
        self.bd_type = bd_type
        self.mode = mode
        self.task_name = task_name

    def __call__(self, num_samples=1):

        return [MetaWorldMT1(bd_type=self.bd_type,
                             max_steps=-1,
                             display=False,
                             assets={},
                             ML_env_name=self.task_name,
                             mode=self.mode,
                             task_id=-1)
                for _ in range(num_samples)]


class SampleSingleExampleFromML10:

    def __init__(self, bd_type, mode):
        self.bd_type = bd_type
        self.mode = mode
        self._ml10obj = None

    def set_ml10obj(self, ml10obj):
        self._ml10obj = ml10obj

    def __call__(self, num_samples):

        task_name = str(
            np.random.choice(list(self._ml10obj.train_classes.keys()))
        ) if self.mode == "train" else str(
            np.random.choice(list(self._ml10obj.test_classes.keys())))
        sample = MetaWorldMT1(bd_type=self.bd_type,
                              max_steps=-1,
                              display=False,
                              assets={},
                              ML_env_name=task_name,
                              mode=self.mode,
                              task_id=-1,
                              ML10_obj=self._ml10obj)

        print(task_name)

        return [sample]


class MetaWorldMT1(Problem):
    """
    Problem based on meta-world MT1. A sample environment will be paired with a random task,
    and at test time only that task will vary. That is, evaluation does not concern
    inter-task transfer but transfer to previously unseen goals (e.g. goal placement, or cube placement 
    in the bin-picking envs

    Note that all experiments (I've checked them one by one to make sure) use the same robot and gripper
    and the same 4d action space and 39d observation space, so we can use the same behavior descriptors
    everywhere.

    Note that (from https://github.com/rlworkgroup/metaworld/issues/65) 
    "The gripper orientation is fixed in a palm-down configuration to make exploration
    easier, so no info on orientation is needed on the observation. All environments are solvable
    without changing the orientation (see the scripted policies in metaworld.policies for proof)"

    So, the 4d action space just corresponds to positions and gripper action.
    """

    def __init__(self,
                 bd_type="type_1",
                 max_steps=-1,
                 display=False,
                 assets={},
                 ML_env_name="pick-place-v2",
                 mode="train",
                 task_id=-1,
                 ML10_obj=None,
                 nb_tasks=1):
        """
        bd_type        The idea is that it fosters exploration related to the task, for now it doesn't matter if it isn't 
                       optimal. Currently those types are available: (noting N the number of samples)
                           - type_0:
                             in R^(3 * N). Only from gripper position, gripper openness/closeness is not taken into account
                             We let the reward take care of supervising the gripper state.
                           - type_1:
                             vector V in R^(4*N). V[:3,i] will be the position of the gripper at the i-th sample, and V[3,i] will
                             be the distance between the two left/right effectors
                           - type_2:
                             vector V in R^(4*N) x C^N where the part in R^4 is the same as type_1, and where C={0,1} indicates 
                             whether the action sent to the gripper is positive or negative (i.e. asking to be open or closed)

        max_steps      this is dicated here by the ML1 tasks, so this argument is ignored and only present for
                       api compatibility.
        display        self explanatory
        assets         ignored, here for api compatibility issues.
        ML_env_name    str, should be one of the strings in metaworld.ML1.ENV_NAMES
        mode           "train" or "test"
        task_id        int. If -1, randomly samples tasks (goal positions), which is what you usually want for normal training/testing.
                            If task_id!=-1, forces the use of the task with the given task_id. That can be useful for visualisation and debug.

        ML10_obj       if None, ML1 is used, otherwise, it should be ML10_obj should be what metaworld.ML10() returns, in which case it will be used to create an env
                       of name ML_env_name with a random task.
        """
        super().__init__()

        self.ML_env_name = ML_env_name
        self.mode = mode

        if ML10_obj is None:
            # note: this is time-consuming, you should probably call that once for all environments, just like you do for ml10
            self.ml1 = metaworld.ML1(
                self.ML_env_name
            )  # constructs the benchmark which is an environment. As this is ML1, only the task (e.g. the goal)
            # will vary (note that in for example pick and place, the initial configuration of the object varies, not the goal).
            # So ml1.train_classes is going to be of lenght 1

            list_sample_class = self.ml1.train_classes if self.mode == "train" else self.ml1.test_classes
            list_sample_task = self.ml1.train_tasks if self.mode == "train" else self.ml1.test_tasks
            
            if nb_tasks == 1:           
                self.env = list_sample_class[self.ML_env_name]()
                self.task_id = np.random.randint(len(
                    list_sample_class)) if task_id == -1 else task_id
                self.task = list_sample_task[self.task_id]  # changes goal
                self.env.set_task(self.task)  # Set task
            else:
                self.env = MultiTaskEnv(list_sample_class, nb_tasks)

        else:
            self.ml10 = ML10_obj
            if self.mode == "train":
                self.env = self.ml10.train_classes[self.ML_env_name]()
                self.task = random.choice([
                    task for task in self.ml10.train_tasks
                    if task.env_name == self.ML_env_name
                ])
                self.task_id = self.ml10.train_tasks.index(self.task)
                self.env.set_task(self.task)  # Set task
            if self.mode == "test":
                self.env = self.ml10.test_classes[self.ML_env_name]()
                self.task = random.choice([
                    task for task in self.ml10.test_tasks
                    if task.env_name == self.ML_env_name
                ])
                self.task_id = self.ml10.test_tasks.index(self.task)
                self.env.set_task(self.task)  # Set task

        if nb_tasks == 1:
            self.dim_obs = self.env.observation_space.shape[
                0]  # in the latest versions of metaworld, it is 39
            self.dim_act = self.env.action_space.shape[
                0]  # should be 4 (end-effector position + grasping activation. there is no orientation)
            self.display = display
        else:
            self.dim_obs = self.env.dim_obs
            self.dim_act = self.env.dim_act
            self.display = False

        self.max_steps = self.env.max_path_length

        self.bd_type = bd_type
        if bd_type == "type_0":  # position only
            self.bd_extractor = class_behavior_descr.GenericBD(
                dims=3, num=2)  # dims*num dimensional
        elif bd_type == "type_1":  # position + gripper effector distances
            self.bd_extractor = class_behavior_descr.GenericBD(
                dims=4, num=2)  # dims*num dimensional
        elif bd_type == "type_2":  # position + gripper effector distances + whether gripper is opening or closing
            self.bd_extractor = class_behavior_descr.GenericBD(
                dims=5, num=2)  # dims*num dimensional
        elif bd_type == "type_3":
            required_dims = self.env._get_pos_objects().shape[
                0]  # in the current state of metaworld, this will either be 3 (for one object) or 6 (two objects)
            self.bd_extractor = class_behavior_descr.GenericBD(
                dims=required_dims,
                num=1)  # final position of manipulated objects
        else:
            raise Exception("Unkown bd type")
        # to avoid adding everyone and their mother to the archive (manually fixed but according to the dimensions of behavior space)
        self.dist_thresh = 0.001
        self.num_saved = 0

    def get_task_info(self):
        """
        useful for keeping track of which agent sovles what in metaworld
        """
        dct = {}
        dct["mode"] = self.mode
        dct["behavior_descriptor_t"] = self.bd_type
        dct["task_id"] = self.task_id
        # dct["global_seed"]=seed_
        problem_consts = {}
        attrs = ["goal", "obj_init_pos", "obj_init_angle"]
        for att in attrs:
            if hasattr(self.env, att):
                problem_consts[att] = getattr(self.env, att)
        dct["problem constant"] = problem_consts
        dct["name"] = self.ML_env_name

        return dct

    def set_env_state(self, state):

        self.env.set_env_state(state)

    def get_end_effector_pos(self):

        return self.env.get_endeff_pos(
        )  # this is as far as I know the same as obs[:3]

    def get_gripper_openness(self):
        """
        attention, this returns 0.1*obs[3]
        """

        dist = self.env._get_site_pos('rightEndEffector') - self.env._get_site_pos(
            'leftEndEffector')
        return np.linalg.norm(
            dist
        )  # as far as I know this is obs[3]/10 (no idea why they multiply it by 10 in obs[3])

    def action_normalisation(self):
        """
        returns a function that should be used as the last non-linearity in agents (to constrain actions in an expected interval). If identity, just return ""
        """
        return "tanh"  # because self.action_space.high is +1, self.action_space.low is -1.

    def close(self):
        self.env.close()

    def get_bd_dims(self):
        return self.bd_extractor.get_bd_dims()

    def get_behavior_space_boundaries(self):
        """
        This is used by br-ns type novelty functions. Not important for meta-world for now
        """
        raise NotImplementedError(
            "not implemented. You can easily get that info from self.env.observation_space.high and .low though."
        )
        return

    def __call__(self, ag, forced_init_state=None, forced_init_obs=None):
        """
        if forced_init_state is not None, self.env.reset() wont be called and the evaluation will start from forced_init_state.
        """

        if hasattr(ag, "eval"):  # in case of torch agent
            ag.eval()

        if self.display:
            self.env.render()

        with torch.no_grad():  # replace with generic context_magager

            if forced_init_state is None:
                obs = self.env.reset()
            else:
                self.set_env_state(forced_init_state)
                obs = self.env._get_obs()

            if forced_init_obs is not None:
                obs = forced_init_obs.copy()

            init_state = self.env.get_env_state()
            init_obs = obs.copy()
            # pdb.set_trace()
            first_action = None

            fitness = 0
            behavior_hist = []
            task_solved = False

            seems_stuck = 0
            stuck_thresh = 1e-5
            prev_eff_pos = self.get_end_effector_pos()

            for it in range(0, self.max_steps):

                if self.display:
                    self.env.render()
                    time.sleep(0.01)

                action = ag(obs)
                if first_action is None:
                    first_action = action
                action = action.flatten().tolist() if isinstance(
                    action, np.ndarray) else action

                obs, reward, done, info = self.env.step(action)

                if self.bd_type == "type_0":
                    behavior_hist.append(obs[:3])
                elif self.bd_type == "type_1":
                    behavior_hist.append(
                        obs[:4]
                    )  # obs[3] is 10*get_gripper_openness(), let's go with that for now
                elif self.bd_type == "type_2":
                    closing_command = float(action[3] > 0)
                    beh = np.zeros(5)
                    beh[:4] = obs[:4].copy()
                    beh[4] = closing_command
                    behavior_hist.append(beh)
                elif self.bd_type == "type_3":
                    # behavior_hist.append(self.env.get_body_com("obj"))
                    behavior_hist.append(
                        self.env._get_pos_objects()
                    )  # this is like calling self.env.get_body_com("obj") or self.env.get_body_com("soccer_ball") etc

                diff_effpos = np.linalg.norm(self.get_end_effector_pos() -
                                             prev_eff_pos)
                if diff_effpos < stuck_thresh:
                    seems_stuck += 1
                else:
                    seems_stuck = 0

                prev_eff_pos = self.get_end_effector_pos()

                if seems_stuck > 80:  # don't set that number too low, since if it doesn't move but closes/opens the gripper it doesn't mena it's stuck
                    task_solved = False
                    done = True

                fitness += reward
                if info["success"]:
                    task_solved = True  # if it is stuck but solves the task, it's okay
                    done = True

                if done:
                    break

            bd = self.bd_extractor.extract_behavior(
                np.array(behavior_hist).reshape(len(behavior_hist),
                                                len(behavior_hist[0])))
            complete_traj = np.concatenate(
                [x.reshape(1, -1) for x in behavior_hist], 0)

        return fitness, bd, task_solved, complete_traj, init_state, init_obs, first_action

    def visualise_bds(self,
                      archive,
                      population,
                      quitely=True,
                      save_to="",
                      generation_num=-1,
                      object_id=0):
        """
        """
        if not quitely and self.display:
            raise Exception(
                "there are some issues with mujoco windows. If self.display, set quitely to True. Otherwise, set self.display to False"
            )

        fig = plt.figure()
        ax = plt.axes(projection='3d', adjustable="box")
        color_population = "red"
        color_archive = "blue"
        # color_closed_gripper="green"
        # color_open_gripper="yellow"
        for x in population:
            bd = x._behavior_descr
            bh = x._complete_trajs
            ax.plot3D(bh[:, 0], bh[:, 1], bh[:, 2], "k--")
            ax.scatter3D(bd[:, 0], bd[:, 1], bd[:, 2], c=color_population)
            # ax.set_aspect('equal') #doesn't support the aspect argument....

        for x in archive:
            bd_a = x._behavior_descr
            bh_a = x._complete_trajs
            ax.plot3D(bh_a[:, 0], bh_a[:, 1], bh_a[:, 2], "m--")
            ax.scatter3D(bd_a[:, 0], bd_a[:, 1], bd_a[:, 2], c=color_archive)
            # ax.set_aspect('equal') #doesn't support the aspect argument....

        if not quitely:
            plt.show()
        else:
            gen_num = generation_num if generation_num != -1 else self.num_saved
            plt.savefig(save_to + f"/gen_{gen_num}.png")
            self.num_saved += 1

        fig.clf()
        plt.close("all")
        gc.collect()
