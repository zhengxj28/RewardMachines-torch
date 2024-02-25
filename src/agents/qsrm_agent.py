import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.networks.rm_network import QRMNet
from src.agents.rm_agent import SRMAgent
from src.agents.base_rl_agent import BaseRLAgent


class QSRMAgent(BaseRLAgent, SRMAgent):
    """
    This class includes a list of policies (a.k.a neural nets) for decomposing reward machines
    """

    def __init__(self, num_features, num_actions, learning_params, model_params, reward_machines, task2rm_id, use_cuda,
                 label_noise):
        SRMAgent.__init__(self, reward_machines, task2rm_id, label_noise)
        BaseRLAgent.__init__(self, use_cuda)

        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_params = learning_params
        self.model_params = model_params
        self.label_noise = label_noise

        device = self.device
        num_policies = self.num_policies  # already defined in RMAgent
        # policy_tran_matrix, terminal already defined in SRMAgent
        self.delta = torch.Tensor(self.policy_tran_matrix).to(device)
        self.srm_done = torch.Tensor(self.terminal_policy_array).to(device)
        self.qrm_net = QRMNet(num_features, num_actions, num_policies, model_params).to(device)
        self.tar_qrm_net = QRMNet(num_features, num_actions, num_policies, model_params).to(device)
        self.buffer = ReplayBuffer(num_features, num_actions, num_policies, self.num_events, learning_params, device)

        if learning_params.tabular_case:
            self.optimizer = optim.SGD(self.qrm_net.parameters(), lr=learning_params.lr)
        else:
            self.optimizer = optim.Adam(self.qrm_net.parameters(), lr=learning_params.lr)

    def learn(self):
        batch_size = self.learning_params.batch_size
        s1, a, s2, rs, nps, _, label_prob, srm_rewards = self.buffer.sample(batch_size)

        ind = torch.LongTensor(range(a.shape[0]))
        Q = self.qrm_net(s1)[ind, :, a.squeeze(1)]
        gamma = self.learning_params.gamma

        with torch.no_grad():
            Q_tar_all = torch.max(self.tar_qrm_net(s2), dim=2)[0]

        loss = torch.Tensor([0.0]).to(self.device)

        if self.learning_params.stochastic_data_augment:
            srm_done = self.srm_done.unsqueeze(0).expand(batch_size, -1)
            Q_tar_all = torch.mul(Q_tar_all, (1-srm_done))
            e_Q_tar = Q_tar_all.unsqueeze(1).unsqueeze(2).expand(-1, self.num_events,
                                                                 self.num_policies, -1)
            e_delta = self.delta.unsqueeze(0).expand(batch_size, -1, -1, -1)
            weighted_Q_tar = torch.mul(e_Q_tar, e_delta).sum(3)

            e_srm_rewards = srm_rewards.unsqueeze(1).expand(-1, self.num_events, -1, -1)
            weighted_rewards = torch.mul(e_srm_rewards, e_delta).sum(3)

            e_label_prob = label_prob.unsqueeze(2).expand(-1, -1, self.num_policies)
            # e_srm_done = srm_done.unsqueeze(1).expand(-1, self.num_events, -1)
            y = torch.mul(e_label_prob, weighted_rewards + gamma * weighted_Q_tar).sum(1)
            for i in range(self.num_policies):
                loss += 0.5 * nn.MSELoss()(Q[:, i], y[:, i])
        else:
            done = torch.zeros_like(nps, device=self.device)
            for terminal_state in self.terminal_policy:
                done[nps == terminal_state] = 1
            Q_tar = torch.gather(Q_tar_all, dim=1, index=nps)
            for i in range(self.num_policies):
                loss += 0.5 * nn.MSELoss()(Q[:, i], rs[:, i] + gamma * Q_tar[:, i] * (1 - done)[:, i])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"value_loss": loss.cpu().item() / self.num_policies}

    def get_action(self, s, eval_mode=False):
        device = self.device
        # belief_policy = self.belief_state2policy(eval_mode)
        belief_policy = self.belief_policy_eval if eval_mode else self.belief_policy
        if not eval_mode and random.random() < self.learning_params.epsilon:
            a = random.choice(range(self.num_actions))
        else:
            with torch.no_grad():
                s = torch.Tensor(s).unsqueeze(0).to(device)
                all_q_value = self.qrm_net(s)
                weight = torch.Tensor(belief_policy).to(device).unsqueeze(0).unsqueeze(-1)
                q_value = torch.mul(all_q_value, weight).sum(1)
                a = torch.argmax(q_value).cpu().item()
        return int(a)

    def update(self, s1, a, s2, info, done, eval_mode=False):
        label_prob = self.get_label_prob(info)
        if not eval_mode:
            # for deterministic data augment
            rewards, next_policies = self.get_rewards_and_next_policies(s1, a, s2, info)
            # for stochastic data augment
            srm_rewards = self.get_srm_rewards(info)
            self.buffer.add_data(s1, a, s2, rewards, next_policies, done, label_prob, srm_rewards)

        # update current rm state
        # self.update_belief_state(info['events'], eval_mode)
        self.update_belief_policy(label_prob, eval_mode)

    def reset_status(self, task, eval_mode=False):
        rm_id = self.task2rm_id[task]
        self.set_rm(rm_id, eval_mode)

    def update_target_network(self):
        self.tar_qrm_net.load_state_dict(self.qrm_net.state_dict())


class ReplayBuffer(object):
    def __init__(self, num_features, num_actions, num_policies, num_events, learning_params, device):
        maxsize = learning_params.buffer_size
        self.maxsize = maxsize
        self.device = device
        self.S1 = torch.empty([maxsize, num_features], device=device)
        self.A = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.S2 = torch.empty([maxsize, num_features], device=device)
        self.Rs = torch.empty([maxsize, num_policies], device=device)
        self.NPs = torch.empty([maxsize, num_policies], dtype=torch.long, device=device)
        self.Done = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        # probability of current label(events)
        self.LabelProb = torch.empty([maxsize, num_events], device=device)
        # rewards of srm, SRMRewards[t,u1,u2]=R(u1,u2)(s_t,a_t,s_{t+1})
        self.SRMRewards = torch.empty([maxsize, num_policies, num_policies], device=device)
        self.index = 0
        self.num_data = 0  # truly stored datas

    def add_data(self, s1, a, s2, rewards, next_policies, done, label_prob, srm_rewards):
        # `rewards[i]` is the reward of policy `i`
        idx = self.index
        self.S1[idx] = torch.Tensor(s1)
        self.A[idx] = torch.LongTensor([a])
        self.S2[idx] = torch.Tensor(s2)
        self.Rs[idx] = torch.Tensor(rewards)
        self.NPs[idx] = torch.LongTensor(next_policies)

        # actually QRM does not use `done` from the environment
        self.Done[idx] = torch.LongTensor([done])

        self.LabelProb[idx] = torch.Tensor(label_prob)
        self.SRMRewards[idx] = torch.Tensor(srm_rewards)

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
        nps = self.NPs[index]
        done = self.Done[index]
        label_prob = self.LabelProb[index]
        srm_rewards = self.SRMRewards[index]
        return s1, a, s2, rs, nps, done, label_prob, srm_rewards
