import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.networks.q_network import DeepQNet
from src.agents.base_rl_agent import BaseRLAgent

class DQNAgent(BaseRLAgent):
    """
    This class includes a list of policies (a.k.a neural nets) for decomposing reward machines
    """

    def __init__(self, num_features, num_actions, learning_params, model_params, use_cuda):
        super().__init__(use_cuda)
        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_params = learning_params

        device = self.device
        self.q_net = DeepQNet(num_features, num_actions, model_params).to(device)
        self.tar_q_net = DeepQNet(num_features, num_actions, model_params).to(device)
        self.buffer = ReplayBuffer(num_features, num_actions, learning_params, device)

        if learning_params.tabular_case:
            self.optimizer = optim.SGD(self.q_net.parameters(), lr=learning_params.lr)
        else:
            self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_params.lr)

    def learn(self):
        s1, a, s2, rs, done = self.buffer.sample(self.learning_params.batch_size)
        # done = torch.zeros_like(nps, device=self.device)
        # done[nps == 0] = 1  # NPs[i]==0 means terminal state

        ind = torch.LongTensor(range(a.shape[0]))
        Q = self.q_net(s1)[ind, a.squeeze(1)]
        gamma = self.learning_params.gamma

        with torch.no_grad():
            Q_tar = torch.max(self.tar_q_net(s2), dim=1)[0]

        loss = torch.Tensor([0.0]).to(self.device)

        rs = rs.squeeze(1)
        done = done.squeeze(1)
        loss += 0.5 * nn.MSELoss()(Q, rs + gamma * Q_tar * (1 - done))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"value_loss": loss.cpu().item()}

    def get_action(self, s, eval_mode=False):
        device = self.device
        if not eval_mode and random.random() < self.learning_params.epsilon:
            a = random.choice(range(self.num_actions))
        else:
            s = torch.Tensor(s).view(1, -1).to(device)
            with torch.no_grad():
                q_value = self.q_net(s)
            a = torch.argmax(q_value).cpu().item()
        return int(a)

    def update(self, s1, a, s2, reward, done, eval_mode=False):
        if not eval_mode:
            self.buffer.add_data(s1, a, s2, reward, done)

    def reset_status(self, *args):
        pass

    def update_target_network(self):
        self.tar_q_net.load_state_dict(self.q_net.state_dict())


class ReplayBuffer(object):
    # TODO: prioritized replay buffer
    def __init__(self, num_features, num_actions, learning_params, device):
        maxsize = learning_params.buffer_size

        self.maxsize = maxsize
        self.device = device
        self.S1 = torch.empty([maxsize, num_features], device=device)
        self.A = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.S2 = torch.empty([maxsize, num_features], device=device)
        self.Rs = torch.empty([maxsize, 1], device=device)
        self.Done = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.index = 0
        self.num_data = 0  # truly stored datas

    def add_data(self, s1, a, s2, reward, done):
        idx = self.index
        self.S1[idx] = torch.Tensor(s1)
        self.A[idx] = torch.LongTensor([a])
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

