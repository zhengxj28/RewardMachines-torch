import torch
import torch.nn as nn
import torch.optim as optim
import random

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
        self.activate_policies = []

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
        for i in range(self.num_policies):
            loss += 0.5 * nn.MSELoss()(Q[:, i], rs[:, i] + gamma * Q_tar[:, i] * (1 - done)[:, i])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"value_loss": loss.cpu().item() / self.num_policies}

    def transfer_knowledge(self):
        if self.learning_params.transfer_methods == "none":
            # simulate learning from scratch, need to re-initialize networks
            self.qrm_net.re_initialize_networks()
        elif self.learning_params.transfer_methods == "equivalent":
            # equivalent states have been merged in a policy when initializing the agent
            # so do nothing here
            pass
        # TODO: implement the following methods
        elif self.learning_params.transfer_methods == "value_com":
            raise NotImplementedError(f"Unknown knowledge transfer methods: {self.learning_params.transfer_methods}")
        elif self.learning_params.transfer_methods == "distill":
            raise NotImplementedError(f"Unknown knowledge transfer methods: {self.learning_params.transfer_methods}")
        else:
            raise NotImplementedError(f"Unknown knowledge transfer methods: {self.learning_params.transfer_methods}")
