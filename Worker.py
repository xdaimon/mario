import numpy as np
import ray, logging
ray.init(logging_level=logging.ERROR)
@ray.remote
class Worker(object):
    def __init__(self, envs_per_worker, worker_id):
        from nes_py.wrappers import JoypadSpace
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
        self.worker_id = worker_id
        self.envs = []
        self.steps = 0
        self.envs_per_worker = envs_per_worker

        for j in range(envs_per_worker):
            self.envs.append(JoypadSpace(gym_super_mario_bros.make('SuperMarioBros-v0'), COMPLEX_MOVEMENT))
            # or use random stage selection
            # self.envs.append(JoypadSpace(gym_super_mario_bros.make('SuperMarioBrosRandomStages-v0'), COMPLEX_MOVEMENT))
            # self.envs[-1].seed(123)

    def step(self, envs_to_step, max_steps):
        self.steps += 1
        results = []
        for env_id,action in envs_to_step:
            env = self.envs[env_id % self.envs_per_worker]
            observation, reward, done, info = env.step(action)
            die = reward < -14
            if done or die or self.steps > max_steps:
                done = True
                observation = env.reset()
            results.append((self.worker_id, env_id, observation.astype(np.float32), reward, done, info, action))
        if self.steps > max_steps:
            self.steps = 0
        return results

    def get_first_observations(self):
        observations = []
        for env in self.envs:
            observations.append(env.reset().astype(np.float32))
        return observations

class Workers:
    def __init__(self, number_workers, envs_per_worker):
        self.ray_ids = []
        self.workers = []
        self.active_envs = []
        for worker_id in range(number_workers):
            self.workers += [Worker.remote(envs_per_worker, worker_id)]
            self.active_envs += [{worker_id*envs_per_worker+j for j in range(envs_per_worker)}]
        self.number_workers = number_workers
        self.envs_per_worker = envs_per_worker

    def get_first_observations(self):
        observations = []
        for worker in self.workers:
            observations += ray.get(worker.get_first_observations.remote())
        return observations

    def set_env_done(self, worker_id, env_id):
        self.active_envs[worker_id].remove(env_id)

    def set_envs_active(self):
        for worker_id in range(self.number_workers):
            self.active_envs[worker_id] = {worker_id*self.envs_per_worker+j for j in range(self.envs_per_worker)}

    def items(self):
        return zip(range(self.number_workers), self.active_envs)

    def step_envs_once(self, worker_id, envs_actions, max_steps):
        self.ray_ids += [self.workers[worker_id].step.remote(envs_actions, max_steps)]

    def results(self):
        for ray_id in self.ray_ids:
            results = ray.get(ray_id)
            yield results
        self.ray_ids = []
