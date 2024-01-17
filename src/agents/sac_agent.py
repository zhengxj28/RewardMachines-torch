import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.networks.ac_network import ActorNetNormalDist, CriticNet
from src.agents.base_rl_agent import BaseRLAgent
from src.common.normalizer import Normalizer, RewardScaler

class SACAgent(BaseRLAgent):
    def __init__(self, num_features, num_actions, action_space, learning_params, model_params, use_cuda):
        super().__init__(use_cuda)
        self.num_features = num_features
        self.num_actions = num_actions
        self.action_space = action_space
        self.learning_params = learning_params
        self.action_bound = model_params.action_bound

        device = self.device
        self.actor_net = ActorNetNormalDist(num_features, num_actions, model_params).to(device)
        self.critic_net1 = CriticNet(num_features + num_actions, 1, model_params).to(device)
        self.critic_net2 = CriticNet(num_features + num_actions, 1, model_params).to(device)
        self.tar_critic_net1 = CriticNet(num_features + num_actions, 1, model_params).to(device)
        self.tar_critic_net2 = CriticNet(num_features + num_actions, 1, model_params).to(device)

        self.buffer = ReplayBuffer(num_features, num_actions, learning_params, device)

        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=learning_params.lr)
        self.critic_optim1 = optim.Adam(self.critic_net1.parameters(), lr=learning_params.lr)
        self.critic_optim2 = optim.Adam(self.critic_net2.parameters(), lr=learning_params.lr)

        # normalizer
        self.state_normalizer = Normalizer(shape=num_features)
        self.reward_normalizer = Normalizer(shape=1)
        self.reward_scaler = RewardScaler(shape=1, gamma=self.learning_params.gamma)
        # use either reward_normalizer or reward_scaler, not both
        assert not (self.learning_params.use_reward_norm and self.learning_params.use_reward_scaling)

        if learning_params.fix_alpha:
            self.alpha = 0.2
        else:
            # adaptive entropy coefficient
            self.target_entropy = -num_actions
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            # learn log_alpha instead of alpha to make sure alpha>0
            self.alpha_optim = optim.Adam([self.log_alpha], lr=learning_params.lr)

    def learn(self):
        s1, a1, s2, r, done = self.buffer.sample(self.learning_params.batch_size)
        gamma = self.learning_params.gamma

        with torch.no_grad():
            a2, log_pi2, _ = self.actor_net(s2)  # a' from the current policy
            # Compute target Q
            target_Q1 = self.tar_critic_net1(torch.cat([s2, a2], 1))
            target_Q2 = self.tar_critic_net2(torch.cat([s2, a2], 1))
            target_Q_soft = r + gamma * (1 - done) * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi2)

        # Compute current Q
        current_Q1 = self.critic_net1(torch.cat([s1, a1], 1))
        current_Q2 = self.critic_net2(torch.cat([s1, a1], 1))
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q_soft) + F.mse_loss(current_Q2, target_Q_soft)
        # Optimize the critic
        self.critic_optim1.zero_grad()
        self.critic_optim2.zero_grad()
        critic_loss.backward()
        self.critic_optim1.step()
        self.critic_optim2.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic_net1.parameters():
            params.requires_grad = False
        for params in self.critic_net2.parameters():
            params.requires_grad = False

        # Compute actor loss
        a1_new, log_pi1, cur_entropy = self.actor_net(s1)
        Q1_nograd = self.critic_net1(torch.cat([s1, a1_new], 1))
        Q2_nograd = self.critic_net2(torch.cat([s1, a1_new], 1))
        Q_soft = torch.min(Q1_nograd, Q2_nograd) - self.alpha * log_pi1
        actor_loss = -Q_soft.mean()

        # Optimize the actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Unfreeze critic networks
        for params in self.critic_net1.parameters():
            params.requires_grad = True
        for params in self.critic_net2.parameters():
            params.requires_grad = True

        # Update alpha
        if not self.learning_params.fix_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(self.log_alpha.exp() * (log_pi1 + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
            report_alpha_loss = alpha_loss.cpu().item()
            report_alpha = self.alpha.cpu().item()
        else:
            report_alpha_loss = 0
            report_alpha = self.alpha

        return {
            "value_loss": critic_loss.cpu().item(),
            "policy_loss": actor_loss.cpu().item(),
            "entropy": torch.mean(cur_entropy).cpu().item(),
            "alpha_loss": report_alpha_loss,
            "alpha": report_alpha
        }

    def get_action(self, s, eval_mode=False, choose_randomly=False):
        if choose_randomly:
            return self.action_space.sample()
        if self.learning_params.use_state_norm:
            s = self.state_normalizer(s, False)
        device = self.device
        s = torch.Tensor(s).view(1, -1).to(device)
        a, _, _ = self.actor_net(s, eval_mode, False)
        return a.cpu().data.numpy().flatten()

    def update(self, s1, a, s2, env_reward, done, eval_mode=False):
        if self.learning_params.use_state_norm:
            # when evaluating, do not update normalizer
            s1 = self.state_normalizer(s1, not eval_mode)
            s2 = self.state_normalizer(s2, not eval_mode)
        if self.learning_params.use_reward_scaling:
            env_reward = self.reward_scaler(env_reward)
        elif self.learning_params.use_reward_norm:
            env_reward = self.reward_normalizer(env_reward, not eval_mode)
        if not eval_mode:
            self.buffer.add_data(s1, a, s2, env_reward, done)

    def update_target_network(self):
        self.soft_update_target_net(self.critic_net1, self.tar_critic_net1)
        self.soft_update_target_net(self.critic_net2, self.tar_critic_net2)

    def soft_update_target_net(self, net1, net2):
        tau = self.learning_params.tau
        for param, target_param in zip(net1.parameters(), net2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class ReplayBuffer(object):
    def __init__(self, num_features, num_actions, learning_params, device):
        maxsize = learning_params.buffer_size
        self.maxsize = maxsize
        self.device = device
        self.S1 = torch.empty([maxsize, num_features], device=device)
        self.A = torch.empty([maxsize, num_actions], device=device)
        self.S2 = torch.empty([maxsize, num_features], device=device)
        self.Rs = torch.empty([maxsize, 1], device=device)
        self.Done = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.index = 0
        self.num_data = 0  # truly stored datas

    def add_data(self, s1, a, s2, reward, done):
        idx = self.index
        self.S1[idx] = torch.Tensor(s1)
        self.A[idx] = torch.Tensor(a)
        self.S2[idx] = torch.Tensor(s2)
        self.Rs[idx] = torch.Tensor([reward])
        self.Done[idx] = torch.LongTensor([done])

        self.index = (self.index + 1) % self.maxsize
        self.num_data = min(self.num_data + 1, self.maxsize)

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        device = self.device
        index = torch.randint(low=0, high=self.num_data, size=[batch_size], device=device)
        s1 = self.S1[index]
        a = self.A[index]
        s2 = self.S2[index]
        rs = self.Rs[index]
        done = self.Done[index]
        return s1, a, s2, rs, done

