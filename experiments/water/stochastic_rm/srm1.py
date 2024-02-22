from src.reward_machines.reward_functions import *
from src.temporal_logic.events_complement import complete_events

num_states = 3
tran_a = [[0, 1, 0],
          [0, 1, 0],
          [0, 0, 1]]
tran_b = [[1, 0, 0],
          [0.1, 0, 0.9],
          [0, 0, 1]]
# tran_ab = [[0, 1, 0],
#            [0.1, 0, 0.9],
#            [0, 0, 1]]
delta_u = {'': [[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]]}
for e in complete_events('a', ['c', 'd', 'e', 'f']):
    delta_u[e] = tran_a
for e in complete_events('b', ['c', 'd', 'e', 'f']):
    delta_u[e] = tran_b

terminal = {2}
pos_terminal = {2}
neg_terminal = {}

r = [[0, 0, 0],
     [0, 0, 1],
     [0, 0, 0]]
reward_matrix = [r]
reward_components = {"constant": 0}

delta_r = [[ConstantRewardFunction(r[i][j]) for j in range(num_states)] for i in range(num_states)]
