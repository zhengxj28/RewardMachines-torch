import torch
import torch.nn as nn
import torch.nn.functional as F


class QRMNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_policies, learning_params):
        super().__init__()
        self.qrm_net = nn.ModuleList()
        self.num_policies = num_policies
        for i in range(num_policies):
            if learning_params.tabular_case:
                q_net = TabularQNet(input_dim, output_dim)
            else:
                q_net = DeepQNet(input_dim, output_dim, learning_params)
            self.qrm_net.append(q_net)

    def forward(self, state):
        # return Q-values of all policies
        q_values = []
        for i in range(self.num_policies):
            q_values.append(self.qrm_net[i](state))
        q_values = torch.stack(q_values, dim=1)
        # dims of q_values: (batch_size, num_policies, num_actions)
        return q_values


class TabularQNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.constant_(self.layer.weight, 1.0)

    def forward(self, state):
        return self.layer(state)


class DeepQNet(nn.Module):
    def __init__(self, input_dim, output_dim, learning_params):
        super().__init__()
        self.layers = nn.ModuleList()
        hidden_dim = learning_params.hidden_dim
        self.num_hidden_layers = learning_params.num_hidden_layers
        for i in range(self.num_hidden_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_dim, hidden_dim))
            elif i < self.num_hidden_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, output_dim))
        # initialize parameters
        for layer in self.layers:
            nn.init.trunc_normal_(layer.weight, std=0.1)
            nn.init.constant_(layer.bias, val=0.1)

    def forward(self, x):
        for i in range(self.num_hidden_layers-1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[self.num_hidden_layers-1](x)
        return x
