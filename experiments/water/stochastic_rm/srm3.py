from src.reward_machines.reward_functions import *
from src.temporal_logic.events_complement import complete_events

"""
go to 'a' first, 
then with probability 0.2 complete the task by 'e', 
with probability 0.8 complete the task by 'f'
"""

num_states = 4
tran_a = [[0, 0.8, 0.2, 0],
          [0, 1, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]
tran_e = [[1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 1]]
tran_f = [[1, 0, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 1, 0],
          [0, 0, 0, 1]]
delta_u = {'': [[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]]
           }

for e in complete_events('a', ['b', 'c', 'd']):
    delta_u[e] = tran_a
for e in complete_events('e', ['b', 'c', 'd']):
    delta_u[e] = tran_e
for e in complete_events('f', ['b', 'c', 'd']):
    delta_u[e] = tran_f

terminal = {3}
pos_terminal = {3}
neg_terminal = {}

r = [[0, 0, 0, 0],
     [0, 0, 0, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]

reward_matrix = [r]
reward_components = {"constant": 0}

delta_r = [[ConstantRewardFunction(r[i][j]) for j in range(num_states)] for i in range(num_states)]

