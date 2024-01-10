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

    def get_true_propositions(self):
        """
        {'reward_ctrl': -0.2433708190917969,
         'reward_run': -0.2302768738318245,
         'x_position': -0.09633745736443508,
         'x_velocity': -0.2302768738318245}
        """
        events = ''
        if self.info['x_position'] < -10:
            events+='n'
        if self.info['x_position'] < -2:
            events+='a'
        if self.info['x_position'] > 2:
            events+='b'
        if self.info['x_position'] > 4:
            events+='c'
        if self.info['x_position'] > 6:
            events+='d'
        if self.info['x_position'] > 8:
            events+='e'
        if self.info['x_position'] > 10:
            events+='f'
        return events


