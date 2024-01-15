import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from src.networks.ac_network import ActorNet, CriticNet
from src.agents.base_rl_agent import BaseRLAgent
from src.common.normalizer import Normalizer, RewardScaler

class PPOAgent(BaseRLAgent):
    def __init__(self, num_features, num_actions, learning_params, model_params, use_cuda, curriculum):
        BaseRLAgent.__init__(self, use_cuda)

        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_params = learning_params
        self.model_params = model_params
        self.action_bound = model_params.action_bound
        self.curriculum = curriculum

        device = self.device

        self.actor_net = ActorNet(num_features, num_actions, model_params).to(device)
        self.critic_net = CriticNet(num_features, 1, model_params).to(device)
        num_policies = 1
        self.buffer = ReplayBuffer(num_features, num_actions, num_policies, learning_params, device)

        self.actor_optim = optim.Adam(self.actor_net.parameters(), lr=learning_params.lr, eps=learning_params.adam_eps)
        self.critic_optim = optim.Adam(self.critic_net.parameters(), lr=learning_params.lr, eps=learning_params.adam_eps)

        # normalizer
        self.state_normalizer = Normalizer(shape=num_features)
        self.reward_normalizer = Normalizer(shape=1)
        self.reward_scaler = RewardScaler(shape=1, gamma=self.learning_params.gamma)
        # use either reward_normalizer or reward_scaler, not both
        assert not (self.learning_params.use_reward_norm and self.learning_params.use_reward_scaling)

    def learn(self):
        s1, a, s2, rs, old_log_prob, done = self.buffer.sample()
        gamma = self.learning_params.gamma
        lam = self.learning_params.lam
        clip_rate = self.learning_params.clip_rate
        p_coef = self.learning_params.policy_loss_coef
        v_coef = self.learning_params.value_loss_coef
        e_coef = self.learning_params.entropy_loss_coef

        ep_len = self.buffer.index
        with torch.no_grad():
            v1_nograd = self.critic_net(s1).squeeze(-1)
            v2 = self.critic_net(s2).squeeze(-1)
            rs = rs.squeeze(-1)
            done = done.squeeze(-1)
            v_tar = rs + gamma * v2 * (1 - done)

            # calculate GAE
            deltas = v_tar - v1_nograd
            gaes = torch.zeros_like(deltas)
            gaes[-1] = deltas[-1]

            for t in range(ep_len - 2, -1, -1):
                next_gaes = gaes[t+1]
                gaes[t] = deltas[t] + gamma * lam * next_gaes
            if self.learning_params.use_adv_norm:
                gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-5)

        loss_dict = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        for _ in range(self.learning_params.n_updates):
            # Random sampling and no repetition.
            # 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for idx in BatchSampler(
                    SubsetRandomSampler(range(ep_len)),
                    self.learning_params.batch_size,
                    False):

                dist = self.actor_net.get_dist(s1[idx])
                new_log_prob = dist.log_prob(a[idx])
                ent = dist.entropy()

                # log_prob (batch, action_dim), we sum the log_prob by action_dim
                ratio = torch.exp(new_log_prob.sum(-1) - old_log_prob[idx].sum(-1))
                surr1 = ratio * gaes[idx]
                surr2 = torch.clamp(ratio, 1 - clip_rate, 1 + clip_rate) * gaes[idx]
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                total_entropy = torch.mean(ent)

                v1 = self.critic_net(s1[idx]).squeeze(-1)

                value_loss = 0.5 * nn.MSELoss()(v1, v_tar[idx])

                loss = p_coef * policy_loss + v_coef * value_loss - e_coef * total_entropy
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                if self.learning_params.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic_net.parameters(), 0.5)
                self.actor_optim.step()
                self.critic_optim.step()

                # record most recent loss
                loss_dict["policy_loss"] = policy_loss.cpu().item()
                loss_dict["value_loss"] = value_loss.cpu().item()
                loss_dict["entropy"] = total_entropy.cpu().item()
        if self.learning_params.use_lr_decay:
            self.lr_decay()
        return loss_dict

    def get_action(self, s, eval_mode=False):
        if self.learning_params.use_state_norm:
            s = self.state_normalizer(s, False)
        device = self.device
        s = torch.Tensor(s).view(1, -1).to(device)
        if not eval_mode:
            with torch.no_grad():
                dist = self.actor_net.get_dist(s)
                a = dist.sample()
                a = torch.clamp(a, -self.action_bound, self.action_bound)
                log_prob = dist.log_prob(a)
            return a.cpu().numpy().flatten(), log_prob
        else:
            with torch.no_grad():
                a, _ = self.actor_net(s)
            return a.cpu().numpy().flatten()

    def update(self, s1, a, s2, env_reward, log_prob, done, eval_mode=False):
        if self.learning_params.use_state_norm:
            # when evaluating, do not update normalizer
            s1 = self.state_normalizer(s1, not eval_mode)
            s2 = self.state_normalizer(s2, not eval_mode)
        if self.learning_params.use_reward_scaling:
            env_reward = self.reward_scaler(env_reward)
        elif self.learning_params.use_reward_norm:
            env_reward = self.reward_normalizer(env_reward, not eval_mode)
        if not eval_mode:
            self.buffer.add_data(s1, a, s2, env_reward, log_prob, done)

    def reset_status(self, task, eval_mode=False):
        if self.learning_params.use_reward_scaling:
            self.reward_scaler.reset()

    def lr_decay(self):
        cur_steps = self.curriculum.get_current_step()
        total_steps = self.curriculum.total_steps
        cur_lr = self.learning_params.lr * (1 - cur_steps/total_steps)
        for p in self.actor_optim.param_groups:
            p['lr'] = cur_lr
        for p in self.critic_optim.param_groups:
            p['lr'] = cur_lr


class ReplayBuffer(object):
    def __init__(self, num_features, num_actions, num_policies, learning_params, device):
        maxsize = learning_params.buffer_size
        assert maxsize >= learning_params.step_unit
        self.maxsize = maxsize
        self.device = device
        self.S1 = torch.empty([maxsize, num_features], device=device)
        # self.A = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.A = torch.empty([maxsize, num_actions], device=device)
        self.S2 = torch.empty([maxsize, num_features], device=device)
        self.R = torch.empty([maxsize, 1], device=device)
        # self.OldLogProb = torch.empty([maxsize, 1], device=device)
        # log_prob of each dimension of continuous-action
        self.OldLogProb = torch.empty([maxsize, num_actions], device=device)
        self.Done = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.index = 0
        self.num_data = 0  # truly stored datas

    def add_data(self, s1, a, s2, rewards, log_prob, done):
        idx = self.index
        self.S1[idx] = torch.Tensor(s1)
        # self.A[idx] = torch.LongTensor([a])
        self.A[idx] = torch.Tensor(a)
        self.S2[idx] = torch.Tensor(s2)
        self.R[idx] = torch.Tensor([rewards])
        self.OldLogProb[idx] = log_prob  # old_prob is a Tensor
        self.Done[idx] = torch.LongTensor([done])

        self.index += 1

    def sample(self):
        s1 = self.S1[:self.index]
        a = self.A[:self.index]
        s2 = self.S2[:self.index]
        r = self.R[:self.index]
        log_prob = self.OldLogProb[:self.index]
        done = self.Done[:self.index]
        return s1, a, s2, r, log_prob, done

    def clear(self):
        self.index = 0

    def is_full(self):
        return self.index >= self.maxsize
