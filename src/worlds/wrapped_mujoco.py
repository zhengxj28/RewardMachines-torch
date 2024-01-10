import gym

class WrappedMujoco(gym.Wrapper):
    def __init__(self, env_name):
        if env_name=="half_cheetah":
            env = gym.make("HalfCheetah-v4")
        else:
            raise ValueError("Unexpected env_name:" + env_name)
        super().__init__(env)
        self.info = {}

    def step(self, action):
        s2, r, done, truncated, info = self.env.step(action)
        self.info = info
        return s2, r, done, truncated, info

    def reset(self, seed=0):
        s0 = self.env.reset(seed=seed)[0]
        self.info = {}
        return s0




