import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.networks.q_network import LTLQNet
from src.agents.base_rl_agent import BaseRLAgent
from src.temporal_logic.ltl_progression import progress, get_propositions
from src.temporal_logic.ltl_preprocess import preprocess, sltl_tokens


class LTLEncDQNAgent(BaseRLAgent):
    """
    This class includes a list of policies (a.k.a neural nets) for decomposing reward machines
    """

    def __init__(self, num_features, num_actions, learning_params, model_params, ltl_formulas, task2rm_id, use_cuda):
        super().__init__(use_cuda)
        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_params = learning_params
        # rm_id equals formula_id
        self.task2formula_id = task2rm_id

        self.ltl_formulas = ltl_formulas
        propositions = set()  # all the propositions occur in ltl_formulas
        for formula in ltl_formulas:
            propositions |= get_propositions(formula)
        # order the set of propositions
        propositions = list(propositions)
        propositions.sort()
        # self.propositions = set(self.propositions)

        self.vocab = dict([(token, i) for i, token in
                           enumerate(sltl_tokens)])  # mapping from ltl token (operations and propositions) to id
        token_id = len(sltl_tokens)
        for p in propositions:
            self.vocab[p] = token_id
            token_id += 1

        # self.init_ltl = None  # initial ltl formula (task specification)
        self.cur_ltl = None  # current ltl formula for training
        self.cur_ltl_eval = None  # current ltl formula for evaluating

        device = self.device
        self.ltl_q_net = LTLQNet(num_features, num_actions, model_params).to(device)
        self.tar_ltl_q_net = LTLQNet(num_features, num_actions, model_params).to(device)
        self.buffer = ReplayBuffer(num_features, num_actions, learning_params, device)

        if learning_params.tabular_case:
            self.optimizer = optim.SGD(self.ltl_q_net.parameters(), lr=learning_params.lr)
        else:
            self.optimizer = optim.Adam(self.ltl_q_net.parameters(), lr=learning_params.lr)

    def learn(self):
        s1, ltl1, a, s2, ltl2, rs, done = self.buffer.sample(self.learning_params.batch_size)
        # done = torch.zeros_like(nps, device=self.device)
        # done[nps == 0] = 1  # NPs[i]==0 means terminal state

        ind = torch.LongTensor(range(a.shape[0]))
        Q = self.ltl_q_net(s1, ltl1)[ind, a.squeeze(1)]
        gamma = self.learning_params.gamma

        with torch.no_grad():
            Q_tar = torch.max(self.tar_ltl_q_net(s2, ltl2), dim=1)[0]

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
            if eval_mode:
                ltl_tensor = self.preprocess_ltl(self.cur_ltl_eval).to(device)
            else:
                ltl_tensor = self.preprocess_ltl(self.cur_ltl).to(device)
            with torch.no_grad():
                q_value = self.ltl_q_net(s, ltl_tensor)
            a = torch.argmax(q_value).cpu().item()
        return int(a)

    def update(self, s1, a, s2, events, done, eval_mode=False):
        # progress ltl formula
        if eval_mode:
            self.cur_ltl_eval = progress(self.cur_ltl_eval, events)
        else:
            reward = 1 if self.cur_ltl == 'True' else 0
            ltl_tensor1 = self.preprocess_ltl(self.cur_ltl)
            self.cur_ltl = progress(self.cur_ltl, events)
            ltl_tensor2 = self.preprocess_ltl(self.cur_ltl)
            self.buffer.add_data(s1, ltl_tensor1, a, s2, ltl_tensor2, reward, done)

    def reset_status(self, task, eval_mode=False):
        formula_id = self.task2formula_id[task]
        if eval_mode:
            self.cur_ltl_eval = self.ltl_formulas[formula_id]
        else:
            self.cur_ltl = self.ltl_formulas[formula_id]

    def update_target_network(self):
        self.tar_ltl_q_net.load_state_dict(self.ltl_q_net.state_dict())

    def preprocess_ltl(self, ltl_formula):
        return preprocess([ltl_formula], self.vocab, self.learning_params.max_ltl_len)


class ReplayBuffer(object):
    # TODO: prioritized replay buffer
    def __init__(self, num_features, num_actions, learning_params, device):
        """
        Create (Prioritized) Replay buffer.
        """
        # self.storage = []

        maxsize = learning_params.buffer_size
        max_ltl_len = learning_params.max_ltl_len

        self.maxsize = maxsize
        self.device = device
        self.S1 = torch.empty([maxsize, num_features], device=device)
        self.LTL1 = torch.empty([maxsize, max_ltl_len], dtype=torch.long, device=device)  # tokenized LTL tensor
        self.A = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.S2 = torch.empty([maxsize, num_features], device=device)
        self.LTL2 = torch.empty([maxsize, max_ltl_len], dtype=torch.long, device=device)
        self.Rs = torch.empty([maxsize, 1], device=device)
        # self.NPs = torch.empty([maxsize, num_policies], dtype=torch.long, device=device)
        self.Done = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.index = 0
        self.num_data = 0  # truly stored datas

    def add_data(self, s1, ltl1, a, s2, ltl2, reward, done):
        # `rewards[i]` is the reward of policy `i`
        idx = self.index
        self.S1[idx] = torch.Tensor(s1)
        self.LTL1[idx] = torch.Tensor(ltl1)
        self.A[idx] = torch.LongTensor([a])
        self.S2[idx] = torch.Tensor(s2)
        self.LTL2[idx] = torch.Tensor(ltl2)
        self.Rs[idx] = torch.Tensor([reward])
        # self.NPs[idx] = torch.LongTensor(next_policies)

        self.Done[idx] = torch.LongTensor([done])

        self.index = (self.index + 1) % self.maxsize
        self.num_data = min(self.num_data + 1, self.maxsize)

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        device = self.device
        index = torch.randint(low=0, high=self.num_data, size=[batch_size], device=device)
        s1 = self.S1[index]
        ltl1 = self.LTL1[index]
        a = self.A[index]
        s2 = self.S2[index]
        ltl2 = self.LTL2[index]
        rs = self.Rs[index]
        done = self.Done[index]
        return s1, ltl1, a, s2, ltl2, rs, done
