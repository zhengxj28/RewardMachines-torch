import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.networks.rm_network import QRMNet
from src.agents.rm_agent import RMAgent
from src.agents.base_rl_agent import BaseRLAgent

class QRMAgent(BaseRLAgent, RMAgent):
    """
    This class includes a list of policies (a.k.a neural nets) for decomposing reward machines
    """

    def __init__(self, num_features, num_actions, learning_params, reward_machines, tester, curriculum, use_cuda):
        RMAgent.__init__(self, reward_machines, tester)

        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_params = learning_params

        self.curriculum = curriculum

        device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
        self.device = device

        num_policies = self.num_policies  # already defined in RMAgent

        self.qrm_net = QRMNet(num_features, num_actions, num_policies, learning_params).to(device)
        self.tar_qrm_net = QRMNet(num_features, num_actions, num_policies, learning_params).to(device)
        self.buffer = ReplayBuffer(num_features, num_actions, num_policies, learning_params, device)

        if learning_params.tabular_case:
            self.optimizer = optim.SGD(self.qrm_net.parameters(), lr=learning_params.lr)
        else:
            self.optimizer = optim.Adam(self.qrm_net.parameters(), lr=learning_params.lr)


    def learn(self):
        s1, a, s2, rs, nps, _ = self.buffer.sample(self.learning_params.batch_size)
        done = torch.zeros_like(nps, device=self.device)
        done[nps == 0] = 1  # NPs[i]==0 means terminal state

        ind = torch.LongTensor(range(a.shape[0]))
        Q = self.qrm_net(s1)[ind, :, a.squeeze(1)]
        gamma = self.learning_params.gamma

        with torch.no_grad():
            Q_tar_all = torch.max(self.tar_qrm_net(s2), dim=2)[0]
            Q_tar = torch.gather(Q_tar_all, dim=1, index=nps)

        loss = torch.Tensor([0.0]).to(self.device)
        for i in range(self.num_policies):
            loss += 0.5 * nn.MSELoss()(Q[:, i], rs[:, i] + gamma * Q_tar[:, i] * (1 - done)[:, i])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"value_loss": loss.cpu().item()}

    def get_action(self, s, eval_mode=False):
        device = self.device
        if eval_mode:
            policy_id = self.state2policy[(self.rm_id_eval, self.u_eval)]
            s = torch.Tensor(s).view(1, -1).to(device)
            q_value = self.qrm_net(s).detach()[:, policy_id].squeeze()
            a = torch.argmax(q_value).cpu().item()
        elif random.random() < self.learning_params.epsilon:
            a = random.choice(range(self.num_actions))
        else:
            policy_id = self.state2policy[(self.rm_id, self.u)]
            s = torch.Tensor(s).view(1, -1).to(device)
            q_value = self.qrm_net(s).detach()[:, policy_id].squeeze()
            a = torch.argmax(q_value).cpu().item()
        return int(a)

    def update(self, s1, a, s2, events, done, eval_mode=False):
        if not eval_mode:
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
            # Adding this experience to the experience replay buffer
            self.buffer.add_data(s1, a, s2, rewards, next_policies, done)

        # update current rm state
        self.update_rm_state(events, eval_mode)

    def reset_status(self, rm_file, eval_mode=False):
        rm_id = self.tester.get_reward_machine_id_from_file(rm_file)
        self.set_rm(rm_id, eval_mode)

    def update_target_network(self):
        self.tar_qrm_net.load_state_dict(self.qrm_net.state_dict())

class ReplayBuffer(object):
    # TODO: prioritized replay buffer
    def __init__(self, num_features, num_actions, num_policies, learning_params, device):
        """
        Create (Prioritized) Replay buffer.
        """
        # self.storage = []

        maxsize = learning_params.buffer_size
        self.maxsize = maxsize
        self.device = device
        self.S1 = torch.empty([maxsize, num_features], device=device)
        self.A = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.S2 = torch.empty([maxsize, num_features], device=device)
        self.Rs = torch.empty([maxsize, num_policies], device=device)
        self.NPs = torch.empty([maxsize, num_policies], dtype=torch.long, device=device)
        self.Done = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.index = 0
        self.num_data = 0  # truly stored datas

        # replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size,
        #                                                                learning_params.prioritized_replay,
        #                                                                learning_params.prioritized_replay_alpha,
        #                                                                learning_params.prioritized_replay_beta0,
        #                                                                curriculum.total_steps if learning_params.prioritized_replay_beta_iters is None else learning_params.prioritized_replay_beta_iters)

    def add_data(self, s1, a, s2, rewards, next_policies, done):
        # `rewards[i]` is the reward of policy `i`
        idx = self.index
        self.S1[idx] = torch.Tensor(s1)
        self.A[idx] = torch.LongTensor([a])
        self.S2[idx] = torch.Tensor(s2)
        self.Rs[idx] = torch.Tensor(rewards)
        self.NPs[idx] = torch.LongTensor(next_policies)

        # actually QRM does not use `done` from the environment
        # self.Done[idx] = torch.LongTensor([done])

        self.index = (self.index + 1) % self.maxsize
        self.num_data = min(self.num_data + 1, self.maxsize)


    def sample(self, batch_size):
        """Sample a batch of experiences."""
        device = self.device
        index = torch.randint(low=0, high=self.num_data, size=[batch_size], device=device)
        # s1 = torch.gather(self.S1, dim=0, index=idxes)
        s1 = self.S1[index]
        a = self.A[index]
        s2 = self.S2[index]
        rs = self.Rs[index]
        nps = self.NPs[index]
        return s1, a, s2, rs, nps, None
