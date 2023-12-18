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
        for i in range(self.num_hidden_layers - 1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[self.num_hidden_layers - 1](x)
        return x


class LTLQNet(nn.Module):
    def __init__(self, num_obs, num_actions, model_params):
        super().__init__()
        self.type = model_params.type
        self.enc_feature = model_params.enc_feature
        if model_params.type == "transformer":
            self.ltl_encoder = TransformerSyn(obs_size=num_obs,
                                              model_params=model_params)
            enc_dim = model_params.d_out
        elif model_params.type == "embedding":
            self.ltl_encoder = nn.Embedding(num_embeddings=model_params.max_num_formulas,
                                            embedding_dim=model_params.embedding_dim)
            enc_dim = model_params.embedding_dim
        else:
            raise NotImplementedError("Unexpected model type:" + model_params.type)

        if model_params.enc_feature == "concat":
            self.q_net = DeepQNet(input_dim=num_obs + enc_dim,
                                  output_dim=num_actions,
                                  model_params=model_params)
        elif model_params.enc_feature == "fine-tune":
            num_neurons = model_params.num_neurons
            self.q_net = DeepQNet(input_dim=num_obs,
                                  output_dim=num_neurons,
                                  model_params=model_params)
            # self.fine_tune_layer = nn.Linear(in_features=model_params.num_neurons+enc_dim,
            #                                  out_features=num_actions)
            self.fine_tune_layer = nn.Sequential(
                nn.Linear(num_neurons+enc_dim, num_neurons),
                nn.ReLU(),
                nn.Linear(num_neurons, num_neurons),
                nn.ReLU(),
                nn.Linear(num_neurons, num_actions),
            )
        else:
            raise NotImplementedError("Unexpected enc_feature process method:" + model_params.enc_feature)

    def forward(self, obs, ltl_input):
        ltl_enc = self.ltl_encoder(ltl_input)
        if self.type == "embedding":
            ltl_enc = ltl_enc.squeeze(1)
        if self.enc_feature == "concat":
            dqn_input = torch.cat([ltl_enc, obs], dim=1)
            q_values = self.q_net(dqn_input)
        elif self.enc_feature == "fine-tune":
            q_hidden = self.q_net(obs)
            q_values = self.fine_tune_layer(torch.cat([ltl_enc, q_hidden], dim=1))
        return q_values
