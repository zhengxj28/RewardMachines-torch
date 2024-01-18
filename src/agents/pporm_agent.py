import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from src.networks.rm_network import ActorRMNet, CriticRMNet
from src.agents.rm_agent import RMAgent
from src.agents.base_rl_agent import BaseRLAgent
from src.common.normalizer import Normalizer, RewardScaler


class PPORMAgent(BaseRLAgent, RMAgent):
    def __init__(self, num_features, num_actions, learning_params, model_params, reward_machines, task2rm_id, use_cuda, curriculum):
        RMAgent.__init__(self, reward_machines, task2rm_id)
        BaseRLAgent.__init__(self, use_cuda)

        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_params = learning_params
        self.model_params = model_params
        self.action_bound = model_params.action_bound
        self.curriculum = curriculum

        device = self.device

        num_policies = self.num_policies  # already defined in RMAgent
        self.actor_rm_net = ActorRMNet(num_features, num_actions, num_policies, model_params).to(device)
        self.critic_rm_net = CriticRMNet(num_features, 1, num_policies, model_params).to(device)
        self.buffer = ReplayBuffer(num_features, num_actions, num_policies, learning_params, device)

        if learning_params.tabular_case:
            raise ValueError("PPO algorithms is not recommended in tabular environment.")
        #     self.actor_optim = optim.Adam(self.actor_rm_net.parameters(), lr=learning_params.lr)
        #     self.critic_optim = optim.SGD(self.critic_rm_net.parameters(), lr=learning_params.lr)
        else:
            self.actor_optim = optim.Adam(self.actor_rm_net.parameters(), lr=learning_params.lr, eps=learning_params.adam_eps)
            self.critic_optim = optim.Adam(self.critic_rm_net.parameters(), lr=learning_params.lr, eps=learning_params.adam_eps)

        # normalizer
        self.state_normalizer = Normalizer(shape=num_features)
        self.reward_normalizer = Normalizer(shape=num_policies)
        self.reward_scaler = RewardScaler(shape=num_policies, gamma=self.learning_params.gamma)
        # use either reward_normalizer or reward_scaler, not both
        assert not (self.learning_params.use_reward_norm and self.learning_params.use_reward_scaling)

    def learn(self):
        s1, a, s2, rs, nps, old_log_prob, _ = self.buffer.sample()
        done = torch.zeros_like(nps, device=self.device)
        done[nps == 0] = 1  # NPs[i]==0 means terminal state
        gamma = self.learning_params.gamma
        lam = self.learning_params.lam
        clip_rate = self.learning_params.clip_rate
        p_coef = self.learning_params.policy_loss_coef
        v_coef = self.learning_params.value_loss_coef
        e_coef = self.learning_params.entropy_loss_coef

        ep_len = self.buffer.index
        with torch.no_grad():
            v1_nograd = self.critic_rm_net(s1).squeeze(-1)
            v2_all = self.critic_rm_net(s2).squeeze(-1)
            v2 = torch.gather(v2_all, dim=1, index=nps)
            v_tar = rs + gamma * v2 * (1 - done)

            # calculate GAEs for each policy u
            deltas = v_tar - v1_nograd
            gaes = torch.zeros_like(deltas)
            gaes[-1, :] = deltas[-1, :]

            for t in range(ep_len - 2, -1, -1):
                next_gaes = torch.gather(gaes[t + 1, :], dim=0, index=nps[t])
                gaes[t, :] = deltas[t, :] + gamma * lam * next_gaes
            if self.learning_params.use_adv_norm:
                gaes = (gaes - gaes.mean(dim=0)) / (gaes.std(dim=0) + 1e-5)

        loss_dict = {"policy_loss": 0, "value_loss": 0, "entropy": 0}
        for _ in range(self.learning_params.n_updates):
            # Random sampling and no repetition.
            # 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for idx in BatchSampler(
                    SubsetRandomSampler(range(ep_len)),
                    self.learning_params.batch_size,
                    False):
                policy_loss = torch.Tensor([0.0]).to(self.device)
                total_entropy = torch.Tensor([0.0]).to(self.device)

                dist_all = self.actor_rm_net.get_dist(s1[idx])  # shape: (batch, policy_id, action)
                for u in range(self.num_policies):
                    dist = dist_all[u]
                    new_log_prob = dist.log_prob(a[idx])
                    ent = dist.entropy()

                    ratio = torch.exp(new_log_prob.sum(-1) - old_log_prob[idx, u].sum(-1))
                    surr1 = ratio * gaes[idx, u]
                    surr2 = torch.clamp(ratio, 1 - clip_rate, 1 + clip_rate) * gaes[idx, u]
                    policy_loss += -torch.mean(torch.min(surr1, surr2))
                    total_entropy += torch.mean(ent)

                v1 = self.critic_rm_net(s1[idx]).squeeze(-1)
                value_loss = torch.Tensor([0.0]).to(self.device)
                for u in range(self.num_policies):
                    value_loss += 0.5 * nn.MSELoss()(v1[:, u], v_tar[idx, u])

                loss = p_coef * policy_loss + v_coef * value_loss - e_coef * total_entropy
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                loss.backward()
                if self.learning_params.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor_rm_net.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic_rm_net.parameters(), 0.5)
                self.actor_optim.step()
                self.critic_optim.step()

                # record most recent loss
                loss_dict["policy_loss"] = policy_loss.cpu().item() / self.num_policies
                loss_dict["value_loss"] = value_loss.cpu().item() / self.num_policies
                loss_dict["entropy"] = total_entropy.cpu().item() / self.num_policies
        if self.learning_params.use_lr_decay:
            self.lr_decay()
        return loss_dict

    def get_action(self, s, eval_mode=False):
        if self.learning_params.use_state_norm:
            s = self.state_normalizer(s, False)
        device = self.device
        s = torch.Tensor(s).view(1, -1).to(device)
        if eval_mode:
            policy_id = self.state2policy[(self.rm_id_eval, self.u_eval)]
        else:
            policy_id = self.state2policy[(self.rm_id, self.u)]
        if not eval_mode:
            with torch.no_grad():
                dist_all = self.actor_rm_net.get_dist(s)
                dist = dist_all[policy_id]
                a = dist.sample()
                a = torch.clamp(a, -self.action_bound, self.action_bound)
                log_prob = torch.stack([dist_all[i].log_prob(a) for i in range(self.num_policies)], dim=1)
            return a.cpu().numpy().flatten(), log_prob
        else:
            with torch.no_grad():
                a = self.actor_rm_net(s)[:, policy_id]
            return a.cpu().numpy().flatten()

    def update(self, s1, a, s2, info, log_prob, done, eval_mode=False):
        if self.learning_params.use_state_norm:
            # when evaluating, do not update normalizer
            s1 = self.state_normalizer(s1, not eval_mode)
            s2 = self.state_normalizer(s2, not eval_mode)

        rewards, next_policies = self.get_rewards_and_next_policies(s1, a, s2, info)
        if self.learning_params.use_reward_scaling:
            rewards = self.reward_scaler(rewards)
        elif self.learning_params.use_reward_norm:
            rewards = self.reward_normalizer(rewards, not eval_mode)
        if not eval_mode:
            self.buffer.add_data(s1, a, s2, rewards, next_policies, log_prob, done)

        # update current rm state
        self.update_rm_state(info['events'], eval_mode)

    def reset_status(self, task, eval_mode=False):
        rm_id = self.task2rm_id[task]
        self.set_rm(rm_id, eval_mode)
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
        self.Rs = torch.empty([maxsize, num_policies], device=device)
        self.NPs = torch.empty([maxsize, num_policies], dtype=torch.long, device=device)
        self.OldLogProb = torch.empty([maxsize, num_policies, num_actions], device=device)
        # self.Done = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.index = 0
        self.num_data = 0  # truly stored datas

    def add_data(self, s1, a, s2, rewards, next_policies, log_prob, done):
        # `rewards[i]` is the reward of policy `i`
        idx = self.index
        self.S1[idx] = torch.Tensor(s1)
        # self.A[idx] = torch.LongTensor([a])
        self.A[idx] = torch.Tensor(a)
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
