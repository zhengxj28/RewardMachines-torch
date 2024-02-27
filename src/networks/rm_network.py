import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks.q_network import TabularQNet, DeepQNet
from src.networks.transformer import TransformerSyn
from src.networks.ac_network import ActorNet, CriticNet, ActorNetNormalDist


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
            probs.append(self.actor_rm_net[i](state)[0])
        probs = torch.stack(probs, dim=1)
        # dims of q_values: (batch_size, num_policies, num_actions)
        return probs

    def get_dist(self, s):
        return [self.actor_rm_net[i].get_dist(s) for i in range(self.num_policies)]


class ActorRMNetNormalDist(nn.Module):
    def __init__(self, num_input, num_output, num_policies, model_params):
        super().__init__()
        self.actor_rm_net = nn.ModuleList(
            [ActorNetNormalDist(num_input, num_output, model_params) for _ in range(num_policies)]
        )
        self.num_policies = num_policies

    def forward(self, state, deterministic=False):
        # return Q-values of all policies
        action_list = []
        log_pi_list = []
        entropy_list = []
        for i in range(self.num_policies):
            a, log_pi, entropy = self.actor_rm_net[i](state, deterministic)
            action_list.append(a)
            log_pi_list.append(log_pi)
            entropy_list.append(entropy)
        all_action = torch.stack(action_list, dim=1)
        all_log_pi = torch.stack(log_pi_list, dim=1)
        all_entropy = torch.stack(entropy_list, dim=1)

        return all_action, all_log_pi, all_entropy


class CriticRMNet(nn.Module):
    def __init__(self, num_input, num_output, num_policies, model_params):
        super().__init__()
        self.critic_rm_net = nn.ModuleList(
            [CriticNet(num_input, num_output, model_params) for _ in range(num_policies)]
        )
        self.num_policies = num_policies

    def forward(self, x):
        values = []
        for i in range(self.num_policies):
            values.append(self.critic_rm_net[i](x))
        values = torch.stack(values, dim=1)
        # dims of values: (batch_size, num_policies, 1)
        return values

    def get_Q_by_all_action(self, s, all_a):
        """
        s.shape=(batch, num_features)
        all_a.shape=(batch, num_policies, num_actions)
        return: all_Q[:,i]=Q(s,i,all_a[:,i])
        """
        all_Q = []
        for i in range(self.num_policies):
            a = all_a[:, i]  # action of policy i
            all_Q.append(self.critic_rm_net[i](torch.cat([s, a], 1)))
        all_Q = torch.stack(all_Q, dim=1)
        return all_Q


class QRMNet(nn.Module):
    # network for the Q-learning for Reward Machines (QRM) algorithm
    def __init__(self, num_obs, num_actions, num_policies, model_params):
        super().__init__()
        self.num_actions = num_actions
        self.num_policies = num_policies
        if model_params.tabular_case:
            self.qrm_net = nn.ModuleList(
                [TabularQNet(num_obs, num_actions) for _ in range(num_policies)]
            )
            self.default_net = TabularQNet(num_obs, num_actions)
        else:
            self.qrm_net = nn.ModuleList(
                [DeepQNet(num_obs, num_actions, model_params) for _ in range(num_policies)]
            )
            self.default_net = DeepQNet(num_obs, num_actions, model_params)
        for param in self.default_net.parameters():
            param.requires_grad = False

    def forward(self, state, partial=False, activate_policies=None, device=None):
        # return Q-values of all policies
        if partial:
            batch_size = state.shape[0]
            q_values = torch.zeros([self.num_policies, batch_size, self.num_actions]).to(device)
            for i in activate_policies:
                q_values[i] = self.qrm_net[i](state)
            q_values = q_values.transpose(0, 1)
        else:
            q_values = []
            for i in range(self.num_policies):
                q_values.append(self.qrm_net[i](state))
            q_values = torch.stack(q_values, dim=1)
        # dims of q_values: (batch_size, num_policies, num_actions)
        return q_values

    def freeze(self, policies):
        for policy in policies:
            for params in self.qrm_net[policy].parameters():
                params.requires_grad = False

    def activate(self, policies):
        for policy in policies:
            for params in self.qrm_net[policy].parameters():
                params.requires_grad = True

    def re_initialize_networks(self):
        for q_net in self.qrm_net:
            q_net.initialize_params()

    def get_param_data_of_policy(self, policy):
        return [param.data for param in self.qrm_net[policy].parameters()]

    def get_default_param_data(self):
        return [param.data for param in self.default_net.parameters()]


class LTLQRMNet(nn.Module):
    def __init__(self, num_obs, num_actions, num_policies, model_params):
        super().__init__()
        self.ltl_encoder = TransformerSyn(obs_size=num_obs,
                                          model_params=model_params)
        enc_dim = model_params.d_out
        self.q_net = QRMNet(num_obs=num_obs + enc_dim,
                            num_actions=num_actions,
                            num_policies=num_policies,
                            model_params=model_params)

    def forward(self, obs, ltl_seq):
        ltl_enc = self.ltl_encoder(ltl_seq)
        dqn_input = torch.cat([ltl_enc, obs])
        q_values = self.q_net(dqn_input)
        return q_values
