import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.networks.rm_network import ActorRMNetNormalDist, CriticRMNet
from src.agents.base_rl_agent import BaseRLAgent
from src.agents.rm_agent import RMAgent
from src.common.normalizer import Normalizer, RewardScaler


class SACRMAgent(BaseRLAgent, RMAgent):
    def __init__(self, num_features, num_actions, action_space, learning_params, model_params, use_cuda,
                 reward_machines, task2rm_id):
        BaseRLAgent.__init__(self, use_cuda)
        RMAgent.__init__(self, reward_machines, task2rm_id)
        self.num_features = num_features
        self.num_actions = num_actions
        self.action_space = action_space
        self.learning_params = learning_params
        self.action_bound = model_params.action_bound

        device = self.device

        num_policies = self.num_policies  # already defined in RMAgent
        self.actor_rm_net = ActorRMNetNormalDist(num_features, num_actions, num_policies, model_params).to(device)
        self.critic_rm_net1 = CriticRMNet(num_features + num_actions, 1, num_policies, model_params).to(device)
        self.critic_rm_net2 = CriticRMNet(num_features + num_actions, 1, num_policies, model_params).to(device)
        self.tar_critic_rm_net1 = CriticRMNet(num_features + num_actions, 1, num_policies, model_params).to(device)
        self.tar_critic_rm_net2 = CriticRMNet(num_features + num_actions, 1, num_policies, model_params).to(device)

        self.buffer = ReplayBuffer(num_features, num_actions, num_policies, learning_params, device)

        self.actor_optim = optim.Adam(self.actor_rm_net.parameters(), lr=learning_params.lr)
        self.critic_optim1 = optim.Adam(self.critic_rm_net1.parameters(), lr=learning_params.lr)
        self.critic_optim2 = optim.Adam(self.critic_rm_net2.parameters(), lr=learning_params.lr)

        # normalizer
        self.state_normalizer = Normalizer(shape=num_features)
        self.reward_normalizer = Normalizer(shape=1)
        self.reward_scaler = RewardScaler(shape=1, gamma=self.learning_params.gamma)
        # use either reward_normalizer or reward_scaler, not both
        assert not (self.learning_params.use_reward_norm and self.learning_params.use_reward_scaling)

        if learning_params.fix_alpha:
            self.alpha = learning_params.init_fixed_alpha
        else:
            # adaptive entropy coefficient
            self.target_entropy = -num_actions
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            # learn log_alpha instead of alpha to make sure alpha>0
            self.alpha_optim = optim.Adam([self.log_alpha], lr=learning_params.lr)

    def learn(self):
        s1, a1, s2, rewards, nps, _ = self.buffer.sample(self.learning_params.batch_size)
        done = torch.zeros_like(nps, device=self.device)
        done[nps == 0] = 1  # NPs[i]==0 means terminal state
        gamma = self.learning_params.gamma
        nps_dim3 = nps.unsqueeze(-1).expand(-1, -1, self.num_actions)

        # update critic
        critic_loss = torch.Tensor([0.0]).to(self.device)
        with torch.no_grad():
            # name `_u1` means tensor[:,i,:] is calculated by policy i
            # name `_u2` means tensor[:,i,:] is calculated by policy nps[:,i]
            a2_u1, log_pi2_u1, _ = self.actor_rm_net(s2)
            # a2_u2[batch, u, :] = a2_u1[batch, nps[batch, u], :]
            # a2_u2 = torch.gather(a2_u1, 1, nps_dim3)
            log_pi2_u1 = log_pi2_u1.squeeze(-1)
            log_pi2_u2 = torch.gather(log_pi2_u1, 1, nps)
            # all_target_Q[:,i]=Q(s2,i,a2), where a2 is sampled from policy i
            all_target_Q1 = self.tar_critic_rm_net1.get_Q_by_all_action(s2, a2_u1).squeeze(-1)
            all_target_Q2 = self.tar_critic_rm_net2.get_Q_by_all_action(s2, a2_u1).squeeze(-1)

        # Compute current Q
        # TODO: a1 sampled from unknown policy
        all_current_Q1 = self.critic_rm_net1(torch.cat([s1, a1], 1)).squeeze(-1)
        all_current_Q2 = self.critic_rm_net2(torch.cat([s1, a1], 1)).squeeze(-1)

        # Compute target Q for each sub-policy
        for i in range(self.num_policies):
            with torch.no_grad():
                log_pi2 = log_pi2_u2[:, i]
                # we use Q(s2,u2,a2) as final target_Q, and u2=nps[u1]
                target_Q1 = torch.gather(all_target_Q1, 1, nps)
                target_Q2 = torch.gather(all_target_Q2, 1, nps)
                target_Q_soft = rewards[:, i] + gamma * (1 - done[:, i]) * (
                            torch.min(target_Q1[:, i], target_Q2[:, i]) - self.alpha * log_pi2)
                target_Q_soft = target_Q_soft.squeeze(-1)
            # Compute critic loss
            critic_loss += F.mse_loss(all_current_Q1[:, i], target_Q_soft) + F.mse_loss(all_current_Q2[:, i],
                                                                                         target_Q_soft)
        # Optimize the critic
        self.critic_optim1.zero_grad()
        self.critic_optim2.zero_grad()
        critic_loss.backward()
        self.critic_optim1.step()
        self.critic_optim2.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic_rm_net1.parameters():
            params.requires_grad = False
        for params in self.critic_rm_net2.parameters():
            params.requires_grad = False

        # Compute actor loss
        actor_loss = torch.Tensor([0.0]).to(self.device)
        all_a1_new, all_log_pi1, all_cur_entropy = self.actor_rm_net(s1)
        all_Q1_nograd = self.critic_rm_net1.get_Q_by_all_action(s1, all_a1_new).squeeze(-1)
        all_Q2_nograd = self.critic_rm_net2.get_Q_by_all_action(s1, all_a1_new).squeeze(-1)
        for i in range(self.num_policies):
            log_pi1 = all_log_pi1[:, i].squeeze(-1)
            Q_soft = torch.min(all_Q1_nograd[:, i], all_Q2_nograd[:, i]) - self.alpha * log_pi1
            actor_loss += -Q_soft.mean()

        # Optimize the actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Unfreeze critic networks
        for params in self.critic_rm_net1.parameters():
            params.requires_grad = True
        for params in self.critic_rm_net2.parameters():
            params.requires_grad = True

        # Update alpha
        if not self.learning_params.fix_alpha:
            pass
            # TODO: adaptive alpha
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            # alpha_loss = -(self.log_alpha.exp() * (log_pi1 + self.target_entropy).detach()).mean()
            # self.alpha_optim.zero_grad()
            # alpha_loss.backward()
            # self.alpha_optim.step()
            # self.alpha = self.log_alpha.exp()
            # report_alpha_loss = alpha_loss.cpu().item()
            # report_alpha = self.alpha.cpu().item()
        else:
            report_alpha_loss = 0
            report_alpha = self.alpha

        return {
            "value_loss": critic_loss.cpu().item() / self.num_policies,
            "policy_loss": actor_loss.cpu().item() / self.num_policies,
            "entropy": all_cur_entropy.mean().cpu().item(),
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
        if eval_mode:
            policy_id = self.state2policy[(self.rm_id_eval, self.u_eval)]
        else:
            policy_id = self.state2policy[(self.rm_id, self.u)]
        all_action, _, _ = self.actor_rm_net(s, eval_mode)
        a = all_action[:, policy_id]
        return a.cpu().data.numpy().flatten()

    def update(self, s1, a, s2, info, done, eval_mode=False):
        rewards, next_policies = self.get_rewards_and_next_policies(s1, a, s2, info)
        # TODO: normalization for rewards
        if self.learning_params.use_state_norm:
            # when evaluating, do not update normalizer
            s1 = self.state_normalizer(s1, not eval_mode)
            s2 = self.state_normalizer(s2, not eval_mode)
        # if self.learning_params.use_reward_scaling:
        #     env_reward = self.reward_scaler(env_reward)
        # elif self.learning_params.use_reward_norm:
        #     env_reward = self.reward_normalizer(env_reward, not eval_mode)
        if not eval_mode:
            self.buffer.add_data(s1, a, s2, rewards, next_policies)

    def update_target_network(self):
        self.soft_update_target_net(self.critic_rm_net1, self.tar_critic_rm_net1)
        self.soft_update_target_net(self.critic_rm_net2, self.tar_critic_rm_net2)

    def soft_update_target_net(self, net1, net2):
        tau = self.learning_params.tau
        for param, target_param in zip(net1.parameters(), net2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def reset_status(self, task, eval_mode=False):
        rm_id = self.task2rm_id[task]
        self.set_rm(rm_id, eval_mode)
        if self.learning_params.use_reward_scaling:
            self.reward_scaler.reset()


class ReplayBuffer(object):
    def __init__(self, num_features, num_actions, num_policies, learning_params, device):
        maxsize = learning_params.buffer_size
        self.maxsize = maxsize
        self.device = device
        self.S1 = torch.empty([maxsize, num_features], device=device)
        self.A = torch.empty([maxsize, num_actions], device=device)
        self.S2 = torch.empty([maxsize, num_features], device=device)
        self.Rs = torch.empty([maxsize, num_policies], device=device)
        self.NPs = torch.empty([maxsize, num_policies], dtype=torch.long, device=device)
        # self.Done = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.index = 0
        self.num_data = 0  # truly stored datas

    def add_data(self, s1, a, s2, rewards, next_policies):
        # `rewards[i]` is the reward of policy `i`
        idx = self.index
        self.S1[idx] = torch.Tensor(s1)
        self.A[idx] = torch.Tensor(a)
        self.S2[idx] = torch.Tensor(s2)
        self.Rs[idx] = torch.Tensor(rewards)
        self.NPs[idx] = torch.LongTensor(next_policies)
        # self.Done[idx] = torch.LongTensor([env_done])
        self.index = (self.index + 1) % self.maxsize
        self.num_data = min(self.num_data + 1, self.maxsize)

    def sample(self, batch_size):
        device = self.device
        index = torch.randint(low=0, high=self.num_data, size=[batch_size], device=device)
        s1 = self.S1[index]
        a = self.A[index]
        s2 = self.S2[index]
        rs = self.Rs[index]
        nps = self.NPs[index]
        # env_done = self.Done[index]
        return s1, a, s2, rs, nps, None
