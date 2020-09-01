from collections import deque
import numpy as np

class FrameStack():
    def __init__(self, env, k):
        self.env = env
        self._k = k
        self._frames = deque([], maxlen=k)
        self.observation_space ={
            "proprioception": env.observation_space["proprioception"],
            "camera": (k,) + env.observation_space["camera"]
        }
        self.action_space = env.action_space

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs[0])
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs[0])
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)
