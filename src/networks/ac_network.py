import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorRMNet(nn.Module):
    def __init__(self, num_input, num_output, num_policies, learning_params):
        super().__init__()
        self.actor_rm_net = nn.ModuleList()
        self.num_policies = num_policies
        for i in range(num_policies):
            if learning_params.tabular_case:
                actor_net = TabularActorNet(num_input, num_output)
            else:
                actor_net = ActorNet(num_input, num_output, learning_params)
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
    def __init__(self, num_input, num_policies, learning_params):
        super().__init__()
        self.critic_rm_net = nn.ModuleList()
        self.num_policies = num_policies
        for i in range(num_policies):
            if learning_params.tabular_case:
                critic_net = TabularCriticNet(num_input)
            else:
                critic_net = CriticNet(num_input, learning_params)
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
        nn.init.constant_(self.layer.weight, 1.0)

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
    def __init__(self, num_input, num_output, learning_params):
        super().__init__()
        self.layers = nn.ModuleList()
        num_neurons = learning_params.num_neurons
        self.num_hidden_layers = learning_params.num_hidden_layers
        for i in range(self.num_hidden_layers):
            if i == 0:
                self.layers.append(nn.Linear(num_input, num_neurons))
            elif i < self.num_hidden_layers - 1:
                self.layers.append(nn.Linear(num_neurons, num_neurons))
            else:
                self.layers.append(nn.Linear(num_neurons, num_output))
        # initialize parameters
        for layer in self.layers:
            nn.init.trunc_normal_(layer.weight, std=0.1)
            nn.init.constant_(layer.bias, val=0.1)

    def forward(self, x):
        for i in range(self.num_hidden_layers-1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[self.num_hidden_layers-1](x)
        # output the probability of choosing each action
        return F.softmax(x, dim=-1)


class CriticNet(nn.Module):
    def __init__(self, num_input, learning_params):
        super().__init__()
        self.layers = nn.ModuleList()
        num_neurons = learning_params.num_neurons
        self.num_hidden_layers = learning_params.num_hidden_layers
        for i in range(self.num_hidden_layers):
            if i == 0:
                self.layers.append(nn.Linear(num_input, num_neurons))
            elif i < self.num_hidden_layers - 1:
                self.layers.append(nn.Linear(num_neurons, num_neurons))
            else:
                self.layers.append(nn.Linear(num_neurons, 1))
        # initialize parameters
        for layer in self.layers:
            nn.init.trunc_normal_(layer.weight, std=0.1)
            nn.init.constant_(layer.bias, val=0.1)

    def forward(self, x):
        for i in range(self.num_hidden_layers-1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[self.num_hidden_layers-1](x)
        # output the value V(s)
        return x
