import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorRMNet(nn.Module):
    def __init__(self, num_input, num_output, num_policies, model_params):
        super().__init__()
        self.actor_rm_net = nn.ModuleList()
        self.num_policies = num_policies
        for i in range(num_policies):
            if model_params.tabular_case:
                actor_net = TabularActorNet(num_input, num_output)
            else:
                actor_net = ActorNet(num_input, num_output, model_params)
            self.actor_rm_net.append(actor_net)

    def forward(self, state):
        # return Q-values of all policies
        probs = []
        for i in range(self.num_policies):
            probs.append(self.actor_rm_net[i](state))
        probs = torch.stack(probs, dim=1)
        # dims of q_values: (batch_size, num_policies, num_actions)
        return probs

class CriticRMNet(nn.Module):
    def __init__(self, num_input, num_policies, model_params):
        super().__init__()
        self.critic_rm_net = nn.ModuleList()
        self.num_policies = num_policies
        for i in range(num_policies):
            if model_params.tabular_case:
                critic_net = TabularCriticNet(num_input)
            else:
                critic_net = CriticNet(num_input, model_params)
            self.critic_rm_net.append(critic_net)

    def forward(self, state):
        # return V-values of all policies
        values = []
        for i in range(self.num_policies):
            values.append(self.critic_rm_net[i](state))
        values = torch.stack(values, dim=1)
        # dims of values: (batch_size, num_policies, 1)
        return values


class TabularActorNet(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        self.layer = nn.Linear(num_input, num_output, bias=False)
        nn.init.normal_(self.layer.weight, mean=0, std=1.0)

    def forward(self, x):
        return F.softmax(self.layer(x), dim=-1)

class TabularCriticNet(nn.Module):
    def __init__(self, num_input):
        super().__init__()
        self.layer = nn.Linear(num_input, 1, bias=False)
        nn.init.constant_(self.layer.weight, 1.0)

    def forward(self, x):
        # output the value V(s)
        return self.layer(x)

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
        # output the probability of choosing each action
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
