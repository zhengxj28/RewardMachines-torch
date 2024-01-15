import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNet(nn.Module):
    def __init__(self, num_input, num_output, model_params):
        super().__init__()
        self.layers = nn.ModuleList()
        num_neurons = model_params.num_neurons
        self.num_hidden_layers = model_params.num_hidden_layers
        self.action_bound = model_params.action_bound
        self.std_module = model_params.std_module
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

        init_log_std = torch.log(model_params.init_std*torch.ones(1, num_output))
        if model_params.std_module=="fixed":
            self.log_std = init_log_std
        elif model_params.std_module=="parameter":
            self.log_std = nn.Parameter(init_log_std)
        elif model_params.std_module=="layer":
            self.log_std_layer = nn.Linear(num_neurons, num_output)
        else:
            raise NotImplementedError(f"std_module {model_params.std_module} not supported.")
        # activation layer
        if model_params.activation=="relu":
            self.activation = nn.ReLU()
        elif model_params.activation=="tanh":
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError(f"Activation layer {model_params.activation} not supported.")

    def forward(self, x):
        for i in range(self.num_hidden_layers-1):
            x = self.layers[i](x)
            x = self.activation(x)
        mean = self.layers[self.num_hidden_layers-1](x)
        mean = nn.Tanh()(mean) * self.action_bound
        if self.std_module in ["fixed", "parameter"]:
            log_std = self.log_std.expand_as(mean)
            std = torch.exp(log_std)
        else:   # layer
            log_std = self.log_std_layer(x)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
        return mean, std

    def get_dist(self, s):
        mean, std = self.forward(s)
        dist = torch.distributions.Normal(mean, std)
        return dist



class CriticNet(nn.Module):
    def __init__(self, num_input, num_output, model_params):
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
                self.layers.append(nn.Linear(num_neurons, num_output))
        # initialize parameters
        for i, layer in enumerate(self.layers):
            # all layers for critic net use gain=1.0
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, val=0)
        # activation layer
        if model_params.activation=="relu":
            self.activation = nn.ReLU()
        elif model_params.activation=="tanh":
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError(f"Activation layer {model_params.activation} not supported.")

    def forward(self, x):
        for i in range(self.num_hidden_layers-1):
            x = self.layers[i](x)
            x = self.activation(x)
        x = self.layers[self.num_hidden_layers-1](x)
        # output the value V(s)
        return x
