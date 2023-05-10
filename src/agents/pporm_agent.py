import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import time, random
import numpy as np
from src.networks.ac_network import ActorRMNet, CriticRMNet
from src.agents.rm_agent import RMAgent


class PPORMAgent(RMAgent):
    """
    This class includes a list of policies (a.k.a neural nets) for decomposing reward machines
    """

    def __init__(self, num_features, num_actions, learning_params, reward_machines, curriculum, use_cuda):
        super().__init__(reward_machines)

        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_params = learning_params

        self.curriculum = curriculum

        device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.device = device

        num_policies = self.num_policies  # already defined in RMAgent

        self.actor_rm_net = ActorRMNet(num_features, num_actions, num_policies, learning_params).to(device)
        self.critic_rm_net = CriticRMNet(num_features, num_policies, learning_params).to(device)
        self.buffer = ReplayBuffer(num_features, num_actions, num_policies, learning_params, device)

        if learning_params.tabular_case:
            self.actor_optim = optim.SGD(self.actor_rm_net.parameters(), lr=learning_params.lr)
            self.critic_optim = optim.SGD(self.critic_rm_net.parameters(), lr=learning_params.lr)
        else:
            self.actor_optim = optim.Adam(self.actor_rm_net.parameters(), lr=learning_params.lr)
            self.critic_optim = optim.Adam(self.critic_rm_net.parameters(), lr=learning_params.lr)

    def learn(self):
        s1, a, s2, rs, nps, old_log_prob, _ = self.buffer.sample()
        done = torch.zeros_like(nps, device=self.device)
        done[nps == 0] = 1  # NPs[i]==0 means terminal state
        gamma = self.learning_params.gamma
        lam = self.learning_params.lam
        clip_rate = self.learning_params.clip_rate

        for _ in range(self.learning_params.n_updates):
            v1 = self.critic_rm_net(s1).squeeze(2)

            with torch.no_grad():
                v2_all = self.critic_rm_net(s2).squeeze(2)
                v2 = torch.gather(v2_all, dim=1, index=nps)
                v_tar = rs + gamma * v2 * (1 - done)

                # calculate gaes for each policy u
                deltas = v_tar - v1
                gaes = torch.zeros_like(deltas)
                gaes[-1, :] = deltas[-1, :]
                ep_len = self.buffer.index
                for t in range(ep_len - 2, -1, -1):
                    gaes[t, :] = deltas[t, :] + gamma * lam * gaes[t+1, :]

            policy_loss = torch.Tensor([0.0]).to(self.device)
            prob_all = self.actor_rm_net(s1)
            for u in range(self.num_policies):
                prob = torch.gather(prob_all[:, u], dim=1, index=a).squeeze(1)
                log_prob = torch.log(prob)
                ratio = torch.exp(log_prob-old_log_prob[:, u])
                surr1 = ratio * gaes[:, u]
                surr2 = torch.clamp(ratio, 1-clip_rate, 1+clip_rate)*gaes[:, u]
                policy_loss += -torch.mean(torch.min(surr1, surr2))

            value_loss = torch.Tensor([0.0]).to(self.device)
            for u in range(self.num_policies):
                value_loss += nn.MSELoss()(v1[:, u], v_tar[:, u])

            loss = policy_loss + value_loss

            self.actor_optim.zero_grad()
            self.critic_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
            self.critic_optim.step()
            return policy_loss.item(), value_loss.item()

    def get_action(self, s, eval_mode=False):
        device = self.device
        policy_id = self.state2policy[(self.rm_id, self.u)]
        s = torch.Tensor(s).view(1, -1).to(device)
        prob_all = self.actor_rm_net(s).detach()  # pi(a|s,u) for all rm state `u`
        prob = prob_all[:, policy_id].squeeze(0)
        dist = Categorical(prob)
        a = dist.sample()
        log_prob = torch.log(prob_all.squeeze(0)[:, a])
        return a.cpu().item(), log_prob

    def update(self, s1, a, s2, events, log_prob, done):
        # Getting rewards and next states for each reward machine to learn
        rewards, next_policies = [], []
        reward_machines = self.reward_machines
        for j in range(len(reward_machines)):
            j_rewards, j_next_states = reward_machines[j].get_rewards_and_next_states(s1, a, s2, events)
            rewards.append(j_rewards)
            next_policies.append(j_next_states)
        # Mapping rewards and next states to specific policies in the policy bank
        rewards = self.map_rewards(rewards)
        next_policies = self.map_next_policies(next_policies)

        # update current rm state
        self.update_rm_state(events)
        # Adding this experience to the experience replay buffer
        self.buffer.add_data(s1, a, s2, rewards, next_policies, log_prob, done)


class ReplayBuffer(object):
    def __init__(self, num_features, num_actions, num_policies, learning_params, device):
        """
        Create (Prioritized) Replay buffer.
        """

        maxsize = learning_params.buffer_size
        self.maxsize = maxsize
        self.device = device
        self.S1 = torch.empty([maxsize, num_features], device=device)
        self.A = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.S2 = torch.empty([maxsize, num_features], device=device)
        self.Rs = torch.empty([maxsize, num_policies], device=device)
        self.NPs = torch.empty([maxsize, num_policies], dtype=torch.long, device=device)
        self.OldLogProb = torch.empty([maxsize, num_policies], device=device)
        # self.Done = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.index = 0
        self.num_data = 0  # truly stored datas

    def add_data(self, s1, a, s2, rewards, next_policies, log_prob, done):
        # `rewards[i]` is the reward of policy `i`
        idx = self.index
        self.S1[idx] = torch.Tensor(s1)
        self.A[idx] = torch.LongTensor([a])
        self.S2[idx] = torch.Tensor(s2)
        self.Rs[idx] = torch.Tensor(rewards)
        self.NPs[idx] = torch.LongTensor(next_policies)
        self.OldLogProb[idx] = log_prob  # old_prob is a Tensor

        # actually PPORM does not use `done` from the environment
        # self.Done[idx] = torch.LongTensor([done])

        self.index += 1

    def sample(self):
        """Sample a (continuous) batch of experiences."""
        s1 = self.S1[:self.index]
        a = self.A[:self.index]
        s2 = self.S2[:self.index]
        rs = self.Rs[:self.index]
        nps = self.NPs[:self.index]
        log_prob = self.OldLogProb[:self.index]
        return s1, a, s2, rs, nps, log_prob, None

    def clear(self):
        self.index = 0

    def is_full(self):
        return self.index >= self.maxsize