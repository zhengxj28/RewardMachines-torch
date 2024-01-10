import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from src.networks.q_network import LTLQNet
from src.agents.base_rl_agent import BaseRLAgent
from src.temporal_logic.ltl_progression import progress, get_propositions, get_progressed_formulas, \
    get_truth_assignments
from src.temporal_logic.ltl_preprocess import preprocess, sltl_tokens


class LTLEncDQNAgent(BaseRLAgent):
    def __init__(self, num_features, num_actions, learning_params, model_params, ltl_formulas, task2rm_id, use_cuda):
        super().__init__(use_cuda)
        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_params = learning_params
        self.model_params = model_params

        # a task is the file path of reward machines or ltl formulas
        # rm_id equals formula_id
        self.task2formula_id = task2rm_id

        self.ltl_formulas = ltl_formulas
        propositions = set()  # all the propositions occur in ltl_formulas
        for formula in ltl_formulas:
            propositions |= get_propositions(formula)
        # order the set of propositions
        propositions = list(propositions)
        propositions.sort()
        self.propositions = propositions
        # self.propositions = set(self.propositions)
        # vocab is a mapping from ltl token (operations and propositions) to id
        self.vocab = dict([(token, i) for i, token in
                           enumerate(sltl_tokens)])
        token_id = len(sltl_tokens)
        for p in propositions:
            self.vocab[p] = token_id
            token_id += 1

        # get a mapping from all possible ltl formulas to id
        possible_ltls = set()
        for formula in ltl_formulas:
            possible_ltls |= get_progressed_formulas(formula)
        possible_ltls = list(possible_ltls)
        possible_ltls.sort(key=lambda x: str(x))
        self.ltl2id = dict([(ltl, i) for i, ltl in enumerate(possible_ltls)])
        assert len(self.ltl2id) <= model_params.max_num_formulas
        self.possible_ltls = possible_ltls

        # get ids of next ltl formula when given events, used for data augmentation
        # next_ltl_id[events][i]==j, means prog(formula_i, events)=formula_j
        label_set = get_truth_assignments(propositions)
        # default ltl_id=-1, in case of unexpected Q-learning update under non-exist formula
        self.next_ltl_ids = dict([(label, -np.ones([model_params.max_num_formulas], dtype=int))
                                  for label in label_set])
        # default reward=0
        self.next_rewards = dict([(label, np.zeros([model_params.max_num_formulas], dtype=int))
                                  for label in label_set])
        for label in label_set:
            for ltl_formula, ltl_id in self.ltl2id.items():
                next_ltl = progress(ltl_formula, label)
                next_ltl_id = self.ltl2id[next_ltl]
                self.next_ltl_ids[label][ltl_id] = next_ltl_id
                self.next_rewards[label][ltl_id] = 1 if next_ltl == 'True' else 0

        self.cur_ltl = None  # current ltl formula for training
        self.cur_ltl_eval = None  # current ltl formula for evaluating

        device = self.device
        self.ltl_q_net = LTLQNet(num_features, num_actions, model_params).to(device)
        self.tar_ltl_q_net = LTLQNet(num_features, num_actions, model_params).to(device)
        self.buffer = ReplayBuffer(num_features, num_actions, learning_params, model_params, device)

        if learning_params.tabular_case:
            self.optimizer = optim.SGD(self.ltl_q_net.parameters(), lr=learning_params.lr)
        else:
            self.optimizer = optim.Adam(self.ltl_q_net.parameters(), lr=learning_params.lr)

    def learn(self):
        s1, ltl1, a, s2, ltl2, reward, next_ltls, next_rewards, done = self.buffer.sample(
            self.learning_params.batch_size)
        # done = torch.zeros_like(nps, device=self.device)
        # done[nps == 0] = 1  # NPs[i]==0 means terminal state


        gamma = self.learning_params.gamma

        if self.learning_params.data_augment:
            # ltl_id>=len(self.ltl2id) means impossible ltl formulas
            # do not use next_ltls[:len(self.ltl2id)] and next_rewards[:len(self.ltl2id)] to learn
            num_ltl = len(self.ltl2id)
            batch_size, num_obs = s1.shape
            s1 = s1.unsqueeze(1).repeat(1, num_ltl, 1)
            s2 = s2.unsqueeze(1).repeat(1, num_ltl, 1)

            cur_ltls = torch.LongTensor(range(num_ltl)).to(self.device).unsqueeze(0).repeat(batch_size, 1)
            next_ltls = next_ltls[:, :num_ltl]
            next_rewards = next_rewards[:, :num_ltl]
            a = a.unsqueeze(1).repeat(1, num_ltl, 1)

            all_dones = torch.zeros_like(next_ltls).to(self.device)
            all_dones[next_ltls == self.ltl2id['False']] = 1
            all_dones[next_ltls == self.ltl2id['True']] = 1

            Q_all = self.ltl_q_net(s1, cur_ltls)
            Q1 = torch.gather(Q_all,-1,a).squeeze(-1)

            with torch.no_grad():
                Q2 = self.tar_ltl_q_net(s2, next_ltls)
                Q_tar = torch.max(Q2, dim=-1)[0]
            loss = 0.5 * nn.MSELoss()(Q1, next_rewards + gamma * Q_tar * (1-all_dones))
        else:
            ind = torch.LongTensor(range(a.shape[0]))
            Q1 = self.ltl_q_net(s1, ltl1)[ind, a.squeeze(1)]
            with torch.no_grad():
                Q_tar = torch.max(self.tar_ltl_q_net(s2, ltl2), dim=1)[0]
            reward = reward.squeeze(1)
            done = done.squeeze(1)
            loss = 0.5 * nn.MSELoss()(Q1, reward + gamma * Q_tar * (1 - done))

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

    def update(self, s1, a, s2, info, done, eval_mode=False):
        events = info['events']
        # progress ltl formula
        if eval_mode:
            self.cur_ltl_eval = progress(self.cur_ltl_eval, events)
        else:
            ltl1 = self.cur_ltl
            ltl_tensor1 = self.preprocess_ltl(ltl1)
            ltl2 = progress(ltl1, events)
            self.cur_ltl = ltl2
            # reward = 1 if ltl2 == 'True' else 0
            ltl_tensor2 = self.preprocess_ltl(ltl2)
            next_ltls = self.get_next_ltls(events)
            next_rewards = self.get_next_rewards(events, s1, a, s2)
            reward = next_rewards[self.ltl2id[ltl1]]
            self.buffer.add_data(s1, ltl_tensor1, a, s2, ltl_tensor2, reward, next_ltls, next_rewards, done)

    def reset_status(self, task, eval_mode=False):
        formula_id = self.task2formula_id[task]
        if eval_mode:
            self.cur_ltl_eval = self.ltl_formulas[formula_id]
        else:
            self.cur_ltl = self.ltl_formulas[formula_id]

    def update_target_network(self):
        self.tar_ltl_q_net.load_state_dict(self.ltl_q_net.state_dict())

    def preprocess_ltl(self, ltl_formula):
        if self.model_params.type == "transformer":
            return preprocess([ltl_formula], self.vocab, self.model_params.max_ltl_len)
        elif self.model_params.type == "embedding":
            assert ltl_formula in self.ltl2id
            return torch.LongTensor([self.ltl2id[ltl_formula]]).unsqueeze(0)

    def get_next_ltls(self, events):
        # for type=="embedding" only
        if events in self.next_ltl_ids:
            return self.next_ltl_ids[events]
        else:
            # default next ltl formula is itself
            return np.array([i for i in range(self.model_params.max_num_formulas)])

    def get_next_rewards(self, events, s1, a, s2):
        if events in self.next_rewards:
            return self.next_rewards[events]
        else:
            return np.zeros([self.model_params.max_num_formulas])


class ReplayBuffer(object):
    # TODO: prioritized replay buffer
    def __init__(self, num_features, num_actions, learning_params, model_params, device):
        maxsize = learning_params.buffer_size
        if model_params.type == "transformer":
            ltl_dim = model_params.max_ltl_len
            if learning_params.data_augment:
                raise NotImplementedError("Data augmentation is not available for model type: \"transformer\".")
        else:
            # store the id of ltl formulas
            ltl_dim = 1
        max_num_formulas = model_params.max_num_formulas

        self.maxsize = maxsize
        self.device = device
        self.S1 = torch.empty([maxsize, num_features], device=device)
        self.LTL1 = torch.empty([maxsize, ltl_dim], dtype=torch.long, device=device)  # tokenized LTL tensor
        self.A = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.S2 = torch.empty([maxsize, num_features], device=device)
        self.LTL2 = torch.empty([maxsize, ltl_dim], dtype=torch.long, device=device)
        self.Rs = torch.empty([maxsize, 1], device=device)
        # NextLTLs and NextRewards are used for data augmentation
        # store the ids of next ltl formulas under current events
        self.NextLTLs = torch.empty([maxsize, max_num_formulas], dtype=torch.long, device=device)
        # store the virtual reward from ltl formula_i to formula_j
        self.NextRewards = torch.empty([maxsize, max_num_formulas], device=device)
        self.Done = torch.empty([maxsize, 1], dtype=torch.long, device=device)
        self.index = 0
        self.num_data = 0  # truly stored datas

    def add_data(self, s1, ltl1, a, s2, ltl2, reward, next_ltls, next_rewards, done):
        # `rewards[i]` is the reward of policy `i`
        idx = self.index
        self.S1[idx] = torch.Tensor(s1)
        self.LTL1[idx] = torch.Tensor(ltl1)
        self.A[idx] = torch.LongTensor([a])
        self.S2[idx] = torch.Tensor(s2)
        self.LTL2[idx] = torch.Tensor(ltl2)
        self.Rs[idx] = torch.Tensor([reward])
        self.NextLTLs[idx] = torch.LongTensor(next_ltls)
        self.NextRewards[idx] = torch.Tensor(next_rewards)
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
        next_ltls = self.NextLTLs[index]
        next_rewards = self.NextRewards[index]
        done = self.Done[index]
        return s1, ltl1, a, s2, ltl2, rs, next_ltls, next_rewards, done
