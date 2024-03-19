import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

from src.agents.qrm_agent import QRMAgent


def power_decrease(start, end, proportion, power=1):
    return start + (end-start)*(proportion**(1/power))

def exponential_decrease(start, end, proportion):
    return start + (end-start)*math.log(proportion*(math.e-1)+1)

class LifelongQRMAgent(QRMAgent):
    """
    This class includes a list of policies (a.k.a neural nets) for decomposing reward machines
    """

    def __init__(self, num_features, num_actions, learning_params, model_params, reward_machines, task2rm_id, use_cuda,
                 curriculum):
        QRMAgent.__init__(self, num_features, num_actions, learning_params, model_params, reward_machines, task2rm_id,
                          use_cuda)
        self.curriculum = curriculum
        lifelong_curriculum = curriculum.lifelong_curriculum
        self.current_phase = 0
        # mark each policy with its corresponding phase
        # to freeze the networks of policies of other phases to simulate "lifelong learning"
        self.policies_of_each_phase = [set() for _ in range(len(lifelong_curriculum))]
        for state_id, policy_id in self.state2policy.items():
            rm_id = state_id[0]
            for phase, tasks_of_phase in enumerate(lifelong_curriculum):
                if rm_id in tasks_of_phase:
                    self.policies_of_each_phase[phase].add(policy_id)
        self.activate_policies = set()
        self.learned_policies = set()

        for com in self.learning_params.value_com:
            assert com in ["average", "max", "left", "right"]

        assert model_params.distill_att in ["none", "n_emb"]
        if model_params.distill_att == "n_emb":
            """
            learn an embedding network for each policy, so there are `n` embedding networks for `n` policies
            each embedding network inputs the environment state, output embedding tensor: emb.shape=(1, num_policies)
            the embedding of learned policy is fixed as a one-hot tensor, emb[i]=1 if the policy_id=i
            we only learn the embedding networks of activated but not learned policies
            the attention for not learned policy `i` is att[i][j]=emb[i]*emb[j]
            the Q-value of policy `i` is estimated as Q_i=\sum_{j\in learned} att[i][j]*Q_j
            """
            from src.networks.attention import NEmbNet
            if learning_params.tabular_case:
                self.n_emb_net = NEmbNet(num_features, self.num_policies, num_hidden_layers=1,
                                         hidden_dim=num_features).to(self.device)

                self.att_optimizer = optim.SGD(self.n_emb_net.parameters(), lr=learning_params.att_lr)
            else:
                self.n_emb_net = NEmbNet(num_features, self.num_policies,
                                         num_hidden_layers=model_params.num_hidden_layers,
                                         hidden_dim=model_params.num_neurons).to(self.device)
                self.att_optimizer = optim.Adam(self.n_emb_net.parameters(), lr=learning_params.att_lr)
            self.tar_n_emb_net = copy.deepcopy(self.n_emb_net)

    def get_action(self, s, eval_mode=False):
        device = self.device
        policy_id = self.state2policy[(self.rm_id_eval, self.u_eval)] \
            if eval_mode else self.state2policy[(self.rm_id, self.u)]
        if not eval_mode and random.random() < self.learning_params.epsilon:
            a = random.choice(range(self.num_actions))
        else:
            s = torch.Tensor(s).view(1, -1).to(device)
            if (eval_mode and self.learning_params.eval_by_student) \
                    or self.model_params.distill_att == "none" \
                    or (not self.learned_policies) \
                    or policy_id in self.learned_policies:
                with torch.no_grad():
                    q_value = self.qrm_net(s, True, {policy_id}, self.device)[:, policy_id].squeeze()
                    a = torch.argmax(q_value).cpu().item()
            else:
                with torch.no_grad():
                    all_q_value = self.qrm_net(s, True, self.learned_policies | {policy_id}, self.device)
                    all_emb = self.n_emb_net(s, self.learned_policies, {policy_id}, self.ltl_correlations)
                    weighted_q = self.calculate_weighted_Q(all_emb, all_q_value)
                    phase_complete_rate = self.curriculum.current_step_of_phase / self.curriculum.phase_total_steps
                    teacher_rate0, teacher_rate1 = self.learning_params.teacher_rates
                    cur_teacher_rate = power_decrease(teacher_rate0, teacher_rate1, phase_complete_rate, self.learning_params.decrease_speed)
                    if random.random() < cur_teacher_rate:
                        final_q = weighted_q[:, policy_id]
                    else:
                        final_q = all_q_value[:, policy_id]
                    # final_q = cur_teacher_rate * weighted_q[:, policy_id] \
                    #           + (1 - cur_teacher_rate) * all_q_value[:, policy_id]
                    a = torch.argmax(final_q.squeeze()).cpu().item()
        return int(a)

    def load_model(self, model_path, learned_task_id):
        for rm_id in learned_task_id:
            for rm_state in self.reward_machines[rm_id].U:
                self.learned_policies.add(self.state2policy[(rm_id, rm_state)])
        file_name = os.path.join(model_path, "qrm_net.pth")
        self.qrm_net.load_state_dict(torch.load(file_name, map_location=f'{self.device.type}:{self.device.index}'))

    def save_model(self, model_path):
        file_name = os.path.join(model_path, "qrm_net.pth")
        torch.save(self.qrm_net.state_dict(), file_name)

    def phase_update(self):
        if self.current_phase > 0 and self.learning_params.transfer_methods!="none":
            for last_learned_policy in self.policies_of_each_phase[self.current_phase - 1]:
                self.learned_policies.add(last_learned_policy)

        # activate networks of current phase and freeze other networks to simulate "lifelong learning"
        # note that a policy may be considered in different phase, due to "equivalent states" of RM
        # must freeze networks of other phases first, then activate networks of current phase
        activate_policies = self.policies_of_each_phase[self.current_phase]
        self.activate_policies = activate_policies
        freeze_policies = set([i for i in range(self.num_policies)]) - activate_policies
        self.qrm_net.freeze(freeze_policies)
        self.qrm_net.activate(activate_policies)
        self.buffer.clear()
        self.current_phase += 1

        # get correlations between activate and learned policies for distillation
        if self.model_params.distill_att == "n_emb":
            self.ltl_correlations = torch.zeros([self.num_policies, self.num_policies], device=self.device)
            w0 = self.learning_params.ltl_correlation_weight
            n = len(self.learned_policies)
            for policy_id in self.activate_policies - self.learned_policies:
                if w0 * n == 0: break
                correlated_policies = self.get_correlated_policies(policy_id, weight=w0)
                k = len(correlated_policies)
                for policy, w in correlated_policies.items():
                    # we hope that exp(bias)=w
                    self.ltl_correlations[policy_id, policy] = math.log(w) if w > 0 else 0

    def get_correlated_policies(self, policy, weight):
        formula = self.policy2ltl[policy]
        if policy in self.learned_policies:
            return {policy: weight}
        elif formula[0] not in ['and', 'or', 'then']:
            return {policy: 0}
        else:
            # it is asserted that formula[1] and formula[2] in progressed formulas
            p1 = self.ltl2policy[formula[1]]
            p2 = self.ltl2policy[formula[2]]
            if formula[0] == 'then':
                p_weight1 = self.get_correlated_policies(p1, weight)
            else:
                p_weight1 = self.get_correlated_policies(p1, weight / 2)
                p_weight2 = self.get_correlated_policies(p2, weight / 2)
                for p, w in p_weight2.items():
                    p_weight1[p] = w
            return p_weight1

    def learn(self):
        loss_info = {}  # loss information to be logged
        s1, a, s2, rs, nps, _ = self.buffer.sample(self.learning_params.batch_size)
        done = torch.zeros_like(nps, device=self.device)
        done[nps == 0] = 1  # NPs[i]==0 means terminal state

        ind = torch.LongTensor(range(a.shape[0]))
        all_Q1 = self.qrm_net(s1, True, self.activate_policies | self.learned_policies, self.device)
        Q = all_Q1[ind, :, a.squeeze(1)]
        gamma = self.learning_params.gamma

        with torch.no_grad():
            Q_tar_all = torch.max(
                self.tar_qrm_net(s2, True, self.activate_policies | self.learned_policies, self.device), dim=2)[0]
            Q_tar = torch.gather(Q_tar_all, dim=1, index=nps)

        num_updated_policies = max(1, len(self.activate_policies - self.learned_policies))
        qrm_loss = torch.Tensor([0.0]).to(self.device)
        for i in self.activate_policies - self.learned_policies:
            qrm_loss += 0.5 * nn.MSELoss()(Q[:, i], rs[:, i] + gamma * Q_tar[:, i] * (1 - done)[:, i])
        loss_info['qrm_loss'] = qrm_loss.cpu().item() / num_updated_policies
        q_net_loss = qrm_loss

        # for policy distillation only
        if self.model_params.distill_att == "n_emb" and self.learned_policies:
            with torch.no_grad():
                all_emb1 = self.n_emb_net(s1, self.learned_policies, self.activate_policies, self.ltl_correlations)
                # note that activate but not learned policies will not be weighted
                w_Q1 = self.calculate_weighted_Q(all_emb1, all_Q1)[ind, :, a.squeeze(1)]
            distill_loss = torch.Tensor([0.0]).to(self.device)
            for i in self.activate_policies - self.learned_policies:
                distill_loss += 0.5 * nn.MSELoss()(Q[:, i], w_Q1[:, i])
                loss_info['distill_loss'] = distill_loss.cpu().item() / num_updated_policies
            phase_complete_rate = self.curriculum.current_step_of_phase / self.curriculum.phase_total_steps
            # adaptive (linear) distill coefficient
            distill_coef0, distill_coef1 = self.learning_params.distill_coefs
            cur_distill_coef = power_decrease(distill_coef0, distill_coef1, phase_complete_rate, self.learning_params.decrease_speed)
            q_net_loss += cur_distill_coef * distill_loss

        self.optimizer.zero_grad()
        q_net_loss.backward()
        self.optimizer.step()

        # update attention networks
        if self.model_params.distill_att == "n_emb" and self.learned_policies:
            with torch.no_grad():
                all_Q1 = self.qrm_net(s1, True, self.learned_policies | self.activate_policies, self.device)
                # current_Q1 = all_Q1[ind, :, a.squeeze(1)]
                all_emb2_tar = self.tar_n_emb_net(s2, self.learned_policies, self.activate_policies,
                                                  self.ltl_correlations)
                all_Q2_tar = self.tar_qrm_net(s2, True, self.learned_policies | self.activate_policies,
                                              self.device)  # (batch, n, a)
                w_Q2_all = torch.max(
                    self.calculate_weighted_Q(all_emb2_tar, all_Q2_tar), dim=2)[0]
                w_Q2 = torch.gather(w_Q2_all, dim=1, index=nps)
            all_emb1 = self.n_emb_net(s1, self.learned_policies, self.activate_policies, self.ltl_correlations)
            w_Q1 = self.calculate_weighted_Q(all_emb1, all_Q1)[ind, :, a.squeeze(1)]

            w_q_loss = torch.Tensor([0.0]).to(self.device)
            # diff_loss = torch.Tensor([0.0]).to(self.device)
            for i in self.activate_policies - self.learned_policies:
                w_q_loss += 0.5 * nn.MSELoss()(w_Q1[:, i], rs[:, i] + gamma * w_Q2[:, i] * (1 - done)[:, i])
                # diff_loss += 0.5 * nn.MSELoss()(w_Q1[:, i], current_Q1[:, i])

            loss_info['w_q_loss'] = w_q_loss.cpu().item() / num_updated_policies
            # loss_info['diff_loss'] = diff_loss.cpu().item() / num_updated_policies

            # w_q_loss_coef = self.learning_params.w_q_loss_coef
            # diff_loss_coef = self.learning_params.diff_loss_coef
            # att_loss = w_q_loss_coef * w_q_loss \
            #            + diff_loss_coef * diff_loss

            att_loss = w_q_loss
            self.att_optimizer.zero_grad()
            att_loss.backward()
            self.att_optimizer.step()
        return loss_info

    def calculate_weighted_Q(self, all_emb, all_Q):
        # all_emb.shape=(batch, n, n), all_Q.shape=(batch, n, a)
        all_emb_expand = all_emb.unsqueeze(3).expand(-1, -1, -1, self.num_actions)
        # tricks: standardize Q-values
        # all_Q = all_Q - all_Q.mean(-1).unsqueeze(-1)
        # assert Q-values of terminal states are 0
        all_Q[:,0]=torch.zeros_like(all_Q[:,0], device=all_Q.device)
        all_Q_expand = all_Q.unsqueeze(1).expand(-1, self.num_policies, -1, -1)
        return torch.mul(all_emb_expand, all_Q_expand).sum(2)

    def transfer_knowledge(self):
        if self.learning_params.transfer_methods == "none":
            # simulate learning from scratch, need to re-initialize networks
            self.qrm_net.re_initialize_networks()
        elif self.learning_params.transfer_methods == "equivalent":
            # equivalent states have been merged into the same policy when initializing the agent,
            # so we do not need to do anything
            pass
        elif self.learning_params.transfer_methods == "value_com":
            for policy in self.activate_policies - self.learned_policies:
                self.transfer_one_policy(policy)
        else:
            raise NotImplementedError(f"Unknown knowledge transfer methods: {self.learning_params.transfer_methods}")

    def transfer_one_policy(self, policy):
        if policy in self.learned_policies:
            return
        # the list of param data of networks
        data_list = self.get_composed_data_list(policy)
        for tar_data, src_data in zip(self.qrm_net.get_param_data_of_policy(policy), data_list):
            tar_data.copy_(src_data)

    def get_composed_data_list(self, policy):
        formula = self.policy2ltl[policy]
        if policy in self.learned_policies:
            data_list = self.qrm_net.get_param_data_of_policy(policy, is_copy=True)
            if self.learning_params.transfer_normalization:
                for i, data in enumerate(data_list):
                    data_list[i] = data - data.mean(0)
            return data_list
        elif formula[0] not in ['and', 'or', 'then']:
            return self.qrm_net.get_default_param_data()
        else:
            # assert formula[1] in self.ltl2policy
            p1 = self.ltl2policy[formula[1]]
            p2 = self.ltl2policy[formula[2]]
            data_l1 = self.get_composed_data_list(p1)
            data_l2 = self.get_composed_data_list(p2)
            # value_com: compose methods for operators "and","or","then" respectively
            value_com = self.learning_params.value_com
            if formula[0] == 'and':
                return self.compose_data(data_l1, data_l2, value_com[0])
            elif formula[0] == 'or':
                return self.compose_data(data_l1, data_l2, value_com[1])
            elif formula[0] == 'then':
                return self.compose_data(data_l1, data_l2, value_com[2])

    def compose_data(self, data_l1, data_l2, compose_method):
        v_composed = []
        for data1, data2 in zip(data_l1, data_l2):
            if compose_method == "average":
                tar_data = (data1 + data2) / 2
            elif compose_method == "max":
                tar_data = torch.max(data1, data2)
            elif compose_method == "left":
                tar_data = data1
            elif compose_method == "right":
                tar_data = data2
            else:
                raise NotImplementedError(f"Unexpected value compose method {compose_method} for knowledge transfer.")
            v_composed.append(tar_data)
        return v_composed

    def update_target_network(self):
        self.tar_qrm_net.load_state_dict(self.qrm_net.state_dict())
        if self.model_params.distill_att == "n_emb":
            self.tar_n_emb_net.load_state_dict(self.n_emb_net.state_dict())

    def update(self, s1, a, s2, info, done, eval_mode=False):
        QRMAgent.update(self, s1, a, s2, info, done, eval_mode)
