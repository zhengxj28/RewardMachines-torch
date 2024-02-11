from src.reward_machines.reward_functions import *
"""
deliver coffee 'f' to office 'g', but need to redeliver coffee with probability 0.1
"""

num_states = 4
delta_u = {'': [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]],
           'f': [[0, 1, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]],
           'g': [[1, 0, 0, 0],
                 [0.1, 0, 0.9, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]],
           'n': [[0, 0, 0, 1],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]]}

terminal = {2, 3}

delta_r = [[ConstantRewardFunction(0), ConstantRewardFunction(0), ConstantRewardFunction(0), ConstantRewardFunction(0)],
           [ConstantRewardFunction(0), ConstantRewardFunction(0), ConstantRewardFunction(1), ConstantRewardFunction(0)],
           [ConstantRewardFunction(0), ConstantRewardFunction(0), ConstantRewardFunction(0), ConstantRewardFunction(0)],
           [ConstantRewardFunction(0), ConstantRewardFunction(0), ConstantRewardFunction(0), ConstantRewardFunction(0)]]
