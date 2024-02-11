from src.reward_machines.reward_functions import *

"""
Adapt from "Inferring Probabilistic Reward Machines from Non-Markovian Reward Signals for Reinforcement Learning"
the coffee machine malfunctions with probability 0.1, 
producing weak coffee which is rejected upon delivery and receives a reward of 0.
"""

num_states = 5
delta_u = {'': [[1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]],
           'f': [[0, 0.9, 0.1, 0, 0],
                 [0, 1, 0, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]],
           'g': [[1, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]],
           'n': [[0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 1],
                 [0, 0, 0, 0, 1],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 1]]}

terminal = {3, 4}

r = [[0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]]

delta_r = [[ConstantRewardFunction(r[i][j]) for j in range(5)] for i in range(5)]
