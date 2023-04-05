import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import time, random
import numpy as np
from src.networks.dqn_network import QRMNet


class QRMAgent:
    """
    This class includes a list of policies (a.k.a neural nets) for decomposing reward machines
    """

    def __init__(self, num_features, num_actions, learning_params, reward_machines, curriculum):
        self.num_actions = num_actions
        self.num_features = num_features
        self.learning_params = learning_params
        self.curriculum = curriculum
        # Decomposing reward machines: We learn one policy per state in a reward machine
        t_i = time.time()
        policies_to_add = self.decompose_reward_machines(reward_machines)
        print("Decomposing RMs is done! (in %0.2f minutes)" % ((time.time() - t_i) / 60))
        self.num_policies = len(policies_to_add)
        self.qrm_net = QRMNet(self.num_features, self.num_actions, self.num_policies)
        self.tar_qrm_net = QRMNet(self.num_features, self.num_actions, self.num_policies)
        self.buffer = ReplayBuffer(learning_params.buffer_size)
        if learning_params.tabular_case:
            self.optimizer = optim.SGD(self.qrm_net.parameters(), lr=learning_params.lr)
        else:
            self.optimizer = optim.Adam(self.qrm_net.parameters(), lr=learning_params.lr)

    def decompose_reward_machines(self, reward_machines):
        self.reward_machines = reward_machines
        # Some machine states might have equivalent Q-functions
        # In those cases, we learn only one policy for them
        policies_to_add = []
        self.state2policy = {}
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

    # def add_policies(self, policies_to_add):
    #     # creating individual networks per policy
    #     num_policies = len(policies_to_add)
    #     if self.learning_params.tabular_case:
    #         return QRMNet(self.num_features, self.num_actions, num_policies)

    # def get_policy(self, rm_id, rm_u):
    #     policy_id = self.state2policy[(rm_id, rm_u)]
    #     return self.policies[policy_id]

    # def get_number_of_policies(self):
    #     return len(self.policies)

    def update_target_network(self):
        self.tar_qrm_net.load_state_dict(self.qrm_net.state_dict())

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

    def learn(self):
        S1, A, S2, Rs, NPs, done = self.buffer.sample(self.learning_params.batch_size)
        S1 = torch.Tensor(S1)
        A = torch.LongTensor(A)
        S2 = torch.Tensor(S2)
        Rs = torch.Tensor(Rs)
        NPs = torch.LongTensor(NPs)
        done = torch.Tensor(done)

        Q = self.qrm_net(S1)[:, :, A]

        # TODO: check Q_tar is correct or not
        Q_tar_all = torch.max(self.tar_qrm_net(S2).detach(), dim=2)[0]
        Q_tar = torch.gather(Q_tar_all, dim=1, index=NPs).unsqueeze(dim=2)
        gamma = self.learning_params.gamma
        loss = nn.MSELoss()(Q, Rs + gamma * Q_tar * (1-done))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_action(self, rm_id, u, s):
        if random.random() < self.learning_params.epsilon:
            a = random.choice(range(self.num_actions))
        else:
            policy_id = self.state2policy[(rm_id, u)]
            s = torch.Tensor(s).view(1, -1)
            q_value = self.qrm_net(s)[:, policy_id].squeeze()
            a = torch.argmax(q_value).item()
        return int(a)


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.storage = []
        self.maxsize = size
        self.index = 0
        # Creating the experience replay buffer
        # replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size,
        #                                                                learning_params.prioritized_replay,
        #                                                                learning_params.prioritized_replay_alpha,
        #                                                                learning_params.prioritized_replay_beta0,
        #                                                                curriculum.total_steps if learning_params.prioritized_replay_beta_iters is None else learning_params.prioritized_replay_beta_iters)

    def __len__(self):
        return len(self.storage)

    def add_data(self, s1, a, s2, rewards, next_policies, done):
        data = (s1, a, s2, rewards, next_policies, done)

        if self.index >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.index] = data
        self.index = (self.index + 1) % self.maxsize

    def _encode_sample(self, idxes):
        S1, A, S2, Rs, NPs, Done = [], [], [], [], [], []
        for i in idxes:
            data = self.storage[i]
            s1, a, s2, rewards, next_policies, done = data
            S1.append(np.array(s1, copy=False))
            A.append(np.array(a, copy=False))
            S2.append(np.array(s2, copy=False))
            Rs.append(np.array(rewards, copy=False))
            NPs.append(np.array(next_policies, copy=False))
            Done.append(np.array(done, copy=False))
        return np.array(S1), np.array(A), np.array(S2), np.array(Rs), np.array(NPs), np.array(Done)

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        idxes = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
