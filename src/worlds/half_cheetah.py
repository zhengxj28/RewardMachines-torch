"""
This code add event detectors to the Ant3 Environment
"""
import gym
import numpy as np
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
# from reward_machines.rm_environment import RewardMachineEnv

class LabellingHalfCheetahEnv:
    def __init__(self):
        # Note that the current position is key for our tasks
        # super().__init__(HalfCheetahEnv(exclude_current_positions_from_observation=False))
        self.env = gym.make("HalfCheetah-v3")


    def step(self, action):
        # executing the action in the environment
        next_obs, original_reward, done, truncated, info = self.env.step(action)
        self.info = info
        env_done = done or truncated
        return next_obs, original_reward, env_done, info

    def get_true_propositions(self):
        events = ''
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


# class MyHalfCheetahEnvRM1(RewardMachineEnv):
#     def __init__(self):
#         env = MyHalfCheetahEnv()
#         rm_files = ["./envs/mujoco_rm/reward_machines/t1.txt"]
#         super().__init__(env, rm_files)
#
# class MyHalfCheetahEnvRM2(RewardMachineEnv):
#     def __init__(self):
#         env = MyHalfCheetahEnv()
#         rm_files = ["./envs/mujoco_rm/reward_machines/t2.txt"]
#         super().__init__(env, rm_files)



if __name__=="__main__":
    env = LabellingHalfCheetahEnv()
    for t in range(10):
        a = env.action_space.sample()
        s2, r, done, info = env.step(a)
        events = env.get_true_propositions()

    print("finish")
