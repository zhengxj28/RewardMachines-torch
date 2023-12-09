import torch
import torch.nn as nn
import torch.nn.functional as F


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
        hidden_dim = learning_params.num_neurons
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
