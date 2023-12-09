import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks.q_network import TabularQNet, DeepQNet
from src.networks.transformer import TransformerSyn

class QRMNet(nn.Module):
    # network for the Q-learning for Reward Machines (QRM) algorithm
    def __init__(self, num_obs, num_actions, num_policies, learning_params):
        super().__init__()
        self.qrm_net = nn.ModuleList()
        self.num_policies = num_policies
        for i in range(num_policies):
            if learning_params.tabular_case:
                q_net = TabularQNet(num_obs, num_actions)
            else:
                q_net = DeepQNet(num_obs, num_actions, learning_params)
            self.qrm_net.append(q_net)

    def forward(self, state):
        # return Q-values of all policies
        q_values = []
        for i in range(self.num_policies):
            q_values.append(self.qrm_net[i](state))
        q_values = torch.stack(q_values, dim=1)
        # dims of q_values: (batch_size, num_policies, num_actions)
        return q_values

class LTLQNet(nn.Module):
    def __init__(self, num_obs, num_actions, learning_params, transformer_params):
        super().__init__()
        self.ltl_encoder = TransformerSyn(obs_size=num_obs,
                                          model_params=transformer_params)
        enc_dim = learning_params.d_out
        self.q_net = DeepQNet(input_dim=num_obs+enc_dim,
                              output_dim=num_actions,
                              learning_params=learning_params)

    def forward(self, obs, ltl_seq):
        ltl_enc = self.ltl_encoder(ltl_seq)
        dqn_input = torch.cat([ltl_enc, obs])
        q_values = self.q_net(dqn_input)
        return q_values

class LTLQRMNet(nn.Module):
    def __init__(self, num_obs, num_actions, num_policies, learning_params):
        super().__init__()
        self.ltl_encoder = TransformerSyn(obs_size=num_obs,
                                          model_params=learning_params)
        enc_dim = learning_params.d_out
        self.q_net = QRMNet(num_obs=num_obs+enc_dim,
                            num_actions=num_actions,
                            num_policies=num_policies,
                            learning_params=learning_params)

    def forward(self, obs, ltl_seq):
        ltl_enc = self.ltl_encoder(ltl_seq)
        dqn_input = torch.cat([ltl_enc, obs])
        q_values = self.q_net(dqn_input)
        return q_values
