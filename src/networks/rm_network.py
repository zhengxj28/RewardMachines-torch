import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks.q_network import TabularQNet, DeepQNet
from src.networks.transformer import TransformerSyn
from src.networks.ac_network import ActorNet, CriticNet


class ActorRMNet(nn.Module):
    def __init__(self, num_input, num_output, num_policies, model_params):
        super().__init__()
        self.actor_rm_net = nn.ModuleList(
            [ActorNet(num_input, num_output, model_params) for _ in range(num_policies)]
        )
        self.num_policies = num_policies

    def forward(self, state):
        # return Q-values of all policies
        probs = []
        for i in range(self.num_policies):
            probs.append(self.actor_rm_net[i](state))
        probs = torch.stack(probs, dim=1)
        # dims of q_values: (batch_size, num_policies, num_actions)
        return probs

    def get_dist(self, s):
        return [self.actor_rm_net[i].get_dist(s) for i in range(self.num_policies)]


class CriticRMNet(nn.Module):
    def __init__(self, num_input, num_policies, model_params):
        super().__init__()
        self.critic_rm_net = nn.ModuleList(
            [CriticNet(num_input, model_params) for _ in range(num_policies)]
        )
        self.num_policies = num_policies

    def forward(self, state):
        # return V-values of all policies
        values = []
        for i in range(self.num_policies):
            values.append(self.critic_rm_net[i](state))
        values = torch.stack(values, dim=1)
        # dims of values: (batch_size, num_policies, 1)
        return values


class QRMNet(nn.Module):
    # network for the Q-learning for Reward Machines (QRM) algorithm
    def __init__(self, num_obs, num_actions, num_policies, model_params):
        super().__init__()
        self.qrm_net = nn.ModuleList()
        self.num_policies = num_policies
        for i in range(num_policies):
            if model_params.tabular_case:
                q_net = TabularQNet(num_obs, num_actions)
            else:
                q_net = DeepQNet(num_obs, num_actions, model_params)
            self.qrm_net.append(q_net)

    def forward(self, state):
        # return Q-values of all policies
        q_values = []
        for i in range(self.num_policies):
            q_values.append(self.qrm_net[i](state))
        q_values = torch.stack(q_values, dim=1)
        # dims of q_values: (batch_size, num_policies, num_actions)
        return q_values



class LTLQRMNet(nn.Module):
    def __init__(self, num_obs, num_actions, num_policies, model_params):
        super().__init__()
        self.ltl_encoder = TransformerSyn(obs_size=num_obs,
                                          model_params=model_params)
        enc_dim = model_params.d_out
        self.q_net = QRMNet(num_obs=num_obs+enc_dim,
                            num_actions=num_actions,
                            num_policies=num_policies,
                            model_params=model_params)

    def forward(self, obs, ltl_seq):
        ltl_enc = self.ltl_encoder(ltl_seq)
        dqn_input = torch.cat([ltl_enc, obs])
        q_values = self.q_net(dqn_input)
        return q_values
