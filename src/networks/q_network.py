import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks.transformer import TransformerSyn

class TabularQNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.constant_(self.layer.weight, 1.0)

    def forward(self, state):
        return self.layer(state)


class DeepQNet(nn.Module):
    def __init__(self, input_dim, output_dim, model_params):
        super().__init__()
        self.layers = nn.ModuleList()
        hidden_dim = model_params.num_neurons
        self.num_hidden_layers = model_params.num_hidden_layers
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

class LTLQNet(nn.Module):
    def __init__(self, num_obs, num_actions, learning_params, transformer_params):
        super().__init__()
        self.ltl_encoder = TransformerSyn(obs_size=num_obs,
                                          model_params=transformer_params)
        enc_dim = learning_params.d_out
        self.q_net = DeepQNet(input_dim=num_obs+enc_dim,
                              output_dim=num_actions,
                              model_params=learning_params)

    def forward(self, obs, ltl_seq):
        ltl_enc = self.ltl_encoder(ltl_seq)
        dqn_input = torch.cat([ltl_enc, obs])
        q_values = self.q_net(dqn_input)
        return q_values