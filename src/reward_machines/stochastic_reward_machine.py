from src.reward_machines.reward_functions import *
# from src.reward_machines.reward_machine_utils import evaluate_dnf, are_these_machines_equivalent, value_iteration
import numpy as np
import torch
from torch.distributions import Categorical
import warnings


class StochasticRewardMachine:
    def __init__(self, file):
        self.generate_by_ltl = False
        path_list = file.split('.')
        module = __import__(file)
        for p in path_list[1:]:
            module = getattr(module, p)
        srm_file = module
        num_states = srm_file.num_states
        self.num_states = num_states
        self.U = [i for i in range(num_states)]  # list of machine states
        self.u0 = 0  # initial state

        self.all_events = list(srm_file.delta_u.keys())
        self.all_events.sort()
        self.events2id = dict([(events, i) for i, events in enumerate(self.all_events)])

        # transition function is represented by 3D array
        # delta_u[l,u1,u2] is the probability of "from u1 to u2 under events l"
        self.delta_u = []
        for events in self.all_events:
            self.delta_u.append(srm_file.delta_u[events])
        # self.delta_u = torch.Tensor(self.delta_u, device="cpu")
        self.delta_u = np.array(self.delta_u)

        # delta_r[u1, u2] is the reward (function) of "from u1 to u2"
        self.reward_matrix = np.array(srm_file.reward_matrix)
        self.reward_components = srm_file.reward_components
        self.delta_r = srm_file.delta_r  # reward-transition function
        self.terminal = srm_file.terminal # set of terminal states (they are automatically detected)
        self.pos_terminal = srm_file.pos_terminal
        self.neg_terminal = srm_file.neg_terminal

    def get_initial_state(self):
        return self.u0

    def get_next_state(self, u1, true_props):
        if true_props in self.events2id:
            events_id = self.events2id[true_props]
            prob_array = self.delta_u[events_id][u1]
            # dist = Categorical(self.delta_u[events_id][u1])
            # u2 = dist.sample().item()
            u2 = np.random.choice(len(prob_array), p=prob_array)
        else:
            # warnings.warn("RM states out of range when getting next state. Set next state `u2=u1` instead")
            u2 = u1
        return u2

    def get_reward(self, u1, u2, s1, a, s2, info, eval_mode=False):
        reward = 0
        if 0 <= u1 < len(self.U) and 0 <= u2 < len(self.U):
            reward += self.delta_r[u1][u2].get_reward(s1, a, s2, info)
        else:
            warnings.warn("RM states out of range when getting reward. Use reward=0 instead.")
        return reward

    def get_rewards_and_next_states(self, s1, a, s2, info):
        rewards = []
        next_states = []
        for u1 in self.U:
            u2 = self.get_next_state(u1, info['events'])
            rewards.append(self.get_reward(u1, u2, s1, a, s2, info))
            next_states.append(u2)
        return rewards, next_states

    def get_states(self):
        return self.U

    def is_terminal_state(self, u1):
        return u1 in self.terminal

    def is_pos_terminal_state(self, u1):
        return u1 in self.pos_terminal

    def is_neg_terminal_state(self, u1):
        return u1 in self.neg_terminal

    def is_this_machine_equivalent(self, u1, rm2, u2):
        return False


if __name__ == "__main__":
    rm = StochasticRewardMachine("experiments.office.stochastic_rm.srm1")
    print()
