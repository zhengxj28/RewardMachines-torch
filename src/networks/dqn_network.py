import torch
import torch.nn as nn
import torch.nn.functional as F


class QRMNet(nn.Module):
    def __init__(self, num_input, num_output, num_policies):
        super().__init__()
        self.qrm_net = nn.ModuleList()
        self.num_policies = num_policies
        for i in range(num_policies):
            layer = nn.Linear(num_input, num_output, bias=False)
            if i == 0:
                nn.init.constant_(layer.weight, 0.0)  # constant policy
            else:
                nn.init.constant_(layer.weight, 1.0)
            self.qrm_net.append(layer)

    def forward(self, state):
        # return Q-values of all policies
        q_values = []
        for i in range(self.num_policies):
            q_values.append(self.qrm_net[i](state))
        q_values = torch.stack(q_values, dim=1)
        # dims of q_values: (batch_size, num_policies, num_actions)
        return q_values

# def create_net(x, num_input, num_output, num_neurons, num_hidden_layers):
#     weights = []
#     for i in range(num_hidden_layers):
#         with tf.variable_scope("Layer_" + str(i)):
#             layer_in  = num_input if i == 0 else num_neurons
#             layer_out = num_output if i == num_hidden_layers-1 else num_neurons
#             add_relu  = i < num_hidden_layers-1
#             x, w = _add_dense_layer(x, layer_in, layer_out, add_relu)
#             weights.extend(w)
#     return x, weights
#
# def create_linear_regression(x, num_input, num_output):
#     W = tf.get_variable("w", [num_input, num_output], initializer=tf.constant_initializer(1.0, dtype=tf.float64), dtype=tf.float64)
#     return tf.matmul(x, W), W
#
# def create_target_updates(weights, target_weights):
#     init_updates = []
#     for i in range(len(weights)):
#         init_updates.append(tf.assign(target_weights[i], weights[i]))
#     return init_updates
#
# def _add_dense_layer(x, num_input, num_output, add_relu):
#     W = tf.get_variable("w", [num_input, num_output], initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float64), dtype=tf.float64)
#     b = tf.get_variable("b", [num_output], initializer=tf.constant_initializer(0.1, dtype=tf.float64), dtype=tf.float64)
#     x = tf.matmul(x, W) + b
#     if add_relu:
#         x = tf.nn.relu(x)
#     return x, [W, b]
