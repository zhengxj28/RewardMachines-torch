import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, num_input, num_output, model_params):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(1, num_output))
        self.layers = nn.ModuleList()
        num_neurons = model_params.num_neurons
        self.num_hidden_layers = model_params.num_hidden_layers
        self.action_bound = model_params.action_bound
        for i in range(self.num_hidden_layers):
            if i == 0:
                self.layers.append(nn.Linear(num_input, num_neurons))
            elif i < self.num_hidden_layers - 1:
                self.layers.append(nn.Linear(num_neurons, num_neurons))
            else:
                self.layers.append(nn.Linear(num_neurons, num_output))
        # initialize parameters
        for i, layer in enumerate(self.layers):
            nn.init.orthogonal_(layer.weight, gain=1.0 if i<len(self.layers)-1 else 0.01)
            nn.init.constant_(layer.bias, val=0)

    def forward(self, x):
        for i in range(self.num_hidden_layers-1):
            x = self.layers[i](x)
            x = nn.Tanh()(x)
        x = self.layers[self.num_hidden_layers-1](x)
        x = nn.Tanh()(x) * self.action_bound
        return x

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        return dist


class CriticNet(nn.Module):
    def __init__(self, num_input, model_params):
        super().__init__()
        self.layers = nn.ModuleList()
        num_neurons = model_params.num_neurons
        self.num_hidden_layers = model_params.num_hidden_layers
        for i in range(self.num_hidden_layers):
            if i == 0:
                self.layers.append(nn.Linear(num_input, num_neurons))
            elif i < self.num_hidden_layers - 1:
                self.layers.append(nn.Linear(num_neurons, num_neurons))
            else:
                self.layers.append(nn.Linear(num_neurons, 1))
        # initialize parameters
        for i, layer in enumerate(self.layers):
            # all layers for critic net use gain=1.0
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, val=0)

    def forward(self, x):
        for i in range(self.num_hidden_layers-1):
            x = self.layers[i](x)
            x = nn.Tanh()(x)
        x = self.layers[self.num_hidden_layers-1](x)
        # output the value V(s)
        return x
