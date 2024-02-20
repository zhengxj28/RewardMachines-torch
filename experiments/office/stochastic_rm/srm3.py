from src.reward_machines.reward_functions import *

"""
go to office 'g' first, 
then with probability 0.8 complete the task by getting coffee 'f', 
with probability 0.2 complete the task by getting mail 'e'
"""

num_states = 5
delta_u = {'': [[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]],
           'e': [[1, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]],
           'f': [[1, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]],
           'g': [[0, 0.8, 0.2, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]],
           'n': [[0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]]}

terminal = {3, 4}
pos_terminal = {3}
neg_terminal = {4}


r = [[0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]]

reward_matrix = [r]
reward_components = {"constant": 0}

delta_r = [[ConstantRewardFunction(r[i][j]) for j in range(5)] for i in range(5)]
