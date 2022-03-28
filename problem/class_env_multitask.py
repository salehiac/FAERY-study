import numpy as np


class MultiTaskEnv:

    def __init__(self, env_gen, nb_tasks):
        
        self.nb_tasks = nb_tasks
        self.envs = [env_gen() for _ in range(self.nb_tasks)]
        self._gen_tasks()

        self.observation_space = nb_tasks * self.envs[0].observation_space.shape[0]
        self.dim_act = nb_tasks * self.envs[0].action_space.shape[0]

        self.max_path_length = self.envs[0].max_path_length

        self.done = self.nb_tasks * [False]
        self.info_success = self.nb_tasks * [False]

        # Useless
        self.goal = None
        self.obj_init_pos = None
        self.obj_init_angle = None
    
    def _gen_tasks(self, from_list):
        copy_from_list = from_list[:]

        for env in self.envs:
            task_id = np.random.randint(len(copy_from_list))
            task = copy_from_list[task_id]
            env.set_task(task)

            copy_from_list = copy_from_list[:task_id] + copy_from_list[task_id+1:]

    def _get_pos_objects(self):
        return np.concatenate([env._get_pos_objects() for env in self.envs])
    
    def get_endeff_pos(self):
        return np.concatenate([env.get_endeff_pos() for env in self.envs])
    
    def close(self):
        for env in self.envs:   env.close()
    
    def reset(self):
        self.done = self.nb_tasks * [False]
        self.info_success = self.nb_tasks * [False]
        return np.concatenate([env.reset() for env in self.envs])

    def _get_obs(self):
        return np.concatenate([env._get_obs() for env in self.envs])

    def get_env_state(self):
        return np.concatenate([env.get_env_state() for env in self.envs])

    def step(self, action):

        obs, reward = np.array([]), 0
        
        for i, env in enumerate(self.envs):

            if self.done[i] is False:

                obs_it, reward_it, done_it, info_it = env.step(action) # CUT ACTION

                obs = np.concatenate([obs, obs_it])
                reward += reward_it
                self.done[i] = done_it or (info_it["success"] is True)

        return obs, reward, all(self.done), {"success":False}
    
    def set_env_state(self, state):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError
        