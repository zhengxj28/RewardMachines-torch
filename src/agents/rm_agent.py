import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import time, random
import numpy as np
from src.networks.q_network import QRMNet


class RMAgent:
    """
    Agent with reward machines only, without RL module.
    """

    def __init__(self, reward_machines):
        self.reward_machines = reward_machines  # all reward machines

        self.rm_id = None  # current reward machine
        self.u = None  # current state of current reward machine
        self.rm_id_eval = None  # current reward machine while evaluating
        self.u_eval = None  # current state of current reward machine while evaluating

        # Decomposing reward machines: We learn one policy per state in a reward machine
        t_i = time.time()
        self.state2policy = {}
        policies_to_add = self.decompose_reward_machines(reward_machines)
        print("Decomposing RMs is done! (in %0.2f minutes)" % ((time.time() - t_i) / 60))
        num_policies = len(policies_to_add)
        self.num_policies = num_policies

    def decompose_reward_machines(self, reward_machines):
        # Some machine states might have equivalent Q-functions
        # In those cases, we learn only one policy for them
        policies_to_add = []
        # We add_data one constant policy for every terminal state
        policies_to_add.append("constant")  # terminal policy has id '0'
        # Associating policies to each machine state
        for i in range(len(reward_machines)):
            rm = reward_machines[i]
            for ui in range(len(rm.get_states())):
                if rm.is_terminal_state(ui):
                    # terminal states goes to the constant policy
                    self.state2policy[(i, ui)] = 0
                else:
                    # associating a policy for this reward machine state
                    policy_id = None
                    for j, uj in self.state2policy:
                        # checking if we already have a policy for an equivalent reward machine
                        if rm.is_this_machine_equivalent(ui, reward_machines[j], uj):
                            print(
                                "Match: reward machine %d from state %d is equivalent to reward machine %d from state "
                                "%d" % (i, ui, j, uj))
                            policy_id = self.state2policy[(j, uj)]
                            break
                    if policy_id is None:
                        # creating a new policy for this node
                        policy_id = len(policies_to_add)
                        policies_to_add.append("machine" + str(i) + "_state" + str(ui))
                    self.state2policy[(i, ui)] = policy_id
        return policies_to_add

    def set_rm(self, rm_id, eval_mode=False):
        if eval_mode:
            self.rm_id_eval = rm_id
            self.u_eval = self.reward_machines[self.rm_id_eval].get_initial_state()
        else:
            self.rm_id = rm_id
            self.u = self.reward_machines[self.rm_id].get_initial_state()


    def map_rewards(self, rewards):
        """
        reward format:
           [R0, ..., Rn] where Ri is the list of rewards gotten by each state on the reward machine 'i'
        returns a single vector with the corresponding rewards given to every policy
        """
        policy_rewards = np.zeros(self.num_policies, dtype=np.float64)
        done = set()
        for i in range(len(rewards)):
            for j in range(len(rewards[i])):
                pos = self.state2policy[(i, j)]
                if pos not in done:
                    policy_rewards[pos] = rewards[i][j]
                    done.add(pos)
                elif policy_rewards[pos] != rewards[i][j]:
                    print("Error! equivalent policies are receiving different rewards!")
                    print("(%d,%d) -> pos %d" % (i, j, pos))
                    print("reward discrepancy:", policy_rewards[pos], "vs", rewards[i][j])
                    print("state2policy", self.state2policy)
                    print("rewards", rewards)
                    exit()
        return policy_rewards

    def map_next_policies(self, next_states):
        """
        next_states format:
           [U0, ..., Un] where Ui is the list of next states for each state on the reward machine 'i'
        returns a single vector with the corresponding next policy per each policy
        """
        next_policies = np.zeros(self.num_policies, dtype=np.float64)
        done = set()
        for i in range(len(next_states)):
            for j in range(len(next_states[i])):
                u = self.state2policy[(i, j)]
                u_next = self.state2policy[(i, next_states[i][j])]
                if u not in done:
                    next_policies[u] = u_next
                    done.add(u)
                elif next_policies[u] != u_next:
                    print("Error! equivalent policies have different next policy!")
                    print("(%d,%d) -> (%d,%d) " % (i, j, u, u_next))
                    print("policy discrepancy:", next_policies[u], "vs", u_next)
                    print("state2policy", self.state2policy)
                    print("next_states", next_states)
                    exit()
        return next_policies

    def update_rm_state(self, events, eval_mode=False):
        if eval_mode:
            self.u_eval = self.reward_machines[self.rm_id_eval].get_next_state(self.u_eval, events)
        else:
            self.u = self.reward_machines[self.rm_id].get_next_state(self.u, events)

