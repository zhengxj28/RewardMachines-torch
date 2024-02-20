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
pos_terminal = {2}
neg_terminal = {3}

r = [[0, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 0, 0, 0],
     [0, 0, 0, 0]]
reward_matrix = [r]
reward_components = {"constant": 0}

delta_r = [[ConstantRewardFunction(r[i][j]) for j in range(4)] for i in range(4)]
