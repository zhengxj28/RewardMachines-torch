import torch
import torch.nn as nn
import torch.optim as optim
import random
import os

from src.agents.qrm_agent import QRMAgent


class LifelongQRMAgent(QRMAgent):
    """
    This class includes a list of policies (a.k.a neural nets) for decomposing reward machines
    """

    def __init__(self, num_features, num_actions, learning_params, model_params, reward_machines, task2rm_id, use_cuda,
                 lifelong_curriculum):
        QRMAgent.__init__(self, num_features, num_actions, learning_params, model_params, reward_machines, task2rm_id,
                          use_cuda)
        self.current_phase = 0
        # mark each policy with its corresponding phase
        # so as to freeze the networks of policies of other phases to simulate "lifelong learning"
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

    def load_model(self, model_path):
        file_name = os.path.join(model_path, "qrm_net.pth")
        self.qrm_net.load_state_dict(torch.load(file_name))

    def save_model(self, model_path):
        file_name = os.path.join(model_path, "qrm_net.pth")
        torch.save(self.qrm_net.state_dict(), file_name)

    def phase_update(self):
        if self.current_phase > 0:
            for last_learned_policy in self.policies_of_each_phase[self.current_phase - 1]:
                self.learned_policies.add(last_learned_policy)

        # activate networks of current phase and freeze other networks to simulate "lifelong learning"
        # note that a policy may considered in different phase, due to "equivalent states" of RM
        # must freeze networks of other phases first, then activate networks of current phase
        activate_policies = self.policies_of_each_phase[self.current_phase]
        self.activate_policies = activate_policies
        freeze_policies = set([i for i in range(self.num_policies)]) - activate_policies
        self.qrm_net.freeze(freeze_policies)
        self.qrm_net.activate(activate_policies)
        self.buffer.clear()
        self.current_phase += 1

    def learn(self):
        s1, a, s2, rs, nps, _ = self.buffer.sample(self.learning_params.batch_size)
        done = torch.zeros_like(nps, device=self.device)
        done[nps == 0] = 1  # NPs[i]==0 means terminal state

        ind = torch.LongTensor(range(a.shape[0]))
        Q = self.qrm_net(s1, True, self.activate_policies, self.device)[ind, :, a.squeeze(1)]
        gamma = self.learning_params.gamma

        with torch.no_grad():
            Q_tar_all = torch.max(
                self.tar_qrm_net(s2, True, self.activate_policies, self.device), dim=2)[0]
            Q_tar = torch.gather(Q_tar_all, dim=1, index=nps)

        # TODO: knowledge distillation methods
        loss = torch.Tensor([0.0]).to(self.device)
        for i in self.activate_policies:
            loss += 0.5 * nn.MSELoss()(Q[:, i], rs[:, i] + gamma * Q_tar[:, i] * (1 - done)[:, i])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"value_loss": loss.cpu().item() / len(self.activate_policies)}

    def transfer_knowledge(self):
        if self.learning_params.transfer_methods == "none":
            # simulate learning from scratch, need to re-initialize networks
            self.qrm_net.re_initialize_networks()
        elif self.learning_params.transfer_methods == "equivalent":
            # equivalent states have been merged into the same policy when initializing the agent
            # so we do not need to do anything
            pass
        # TODO: implement the following methods
        elif self.learning_params.transfer_methods == "value_com":
            for policy in self.activate_policies:
                self.transfer_one_policy(policy)
        elif self.learning_params.transfer_methods == "distill":
            raise NotImplementedError(f"Unknown knowledge transfer methods: {self.learning_params.transfer_methods}")
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
