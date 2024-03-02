import torch
import torch.nn as nn
import torch.nn.functional as F
from src.networks.transformer import TransformerSyn


class NEmbNet(nn.Module):
    def __init__(self, num_features, num_policies, num_hidden_layers, hidden_dim):
        super().__init__()
        self.num_policies = num_policies
        self.n_emb = nn.ModuleList(
            [OneEmbNet(num_features, num_policies, num_hidden_layers, hidden_dim) for _ in range(num_policies)]
        )

    def forward(self, state, learned_policies, activate_policies):
        learned_idx = torch.LongTensor(list(learned_policies)).to(state.device)
        activate_idx = list(activate_policies)
        # not need to forward emb_net for learned_policies
        # all_emb = torch.zeros([state.shape[0], self.num_policies, self.num_policies], device=state.device)
        all_emb = torch.eye(self.num_policies, device=state.device).repeat(state.shape[0],1,1)
        for i in activate_policies-learned_policies:
            all_emb[:, i] = self.n_emb[i](state, learned_idx)
        return all_emb


class OneEmbNet(nn.Module):
    def __init__(self, num_features, num_policies, num_hidden_layers, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_hidden_layers = num_hidden_layers
        for i in range(num_hidden_layers):
            if num_hidden_layers > 1 and i == 0:
                self.layers.append(nn.Linear(num_features, hidden_dim))
            elif i < self.num_hidden_layers - 1:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                # softmax for the last layer, so do not need bias
                self.layers.append(nn.Linear(hidden_dim, num_policies, bias=False))
        # TODO: initialize networks

    def forward(self, x, learned_idx):
        for i in range(self.num_hidden_layers - 1):
            x = self.layers[i](x)
            x = F.relu(x)
        x = self.layers[self.num_hidden_layers - 1](x)
        sub_emb = torch.softmax(x[:, learned_idx], dim=1)
        emb = torch.zeros_like(x, device=x.device)
        emb[:, learned_idx] = sub_emb
        return emb