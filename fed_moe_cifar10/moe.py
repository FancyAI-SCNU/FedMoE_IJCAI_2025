# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
import torch.nn.functional as F
from model.resnet import ResNet18

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, alpha=1, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)*alpha
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # # add eps to all zero values in order to avoid nans when going back to log space
        # combined[combined == 0] = np.finfo(float).eps
        # # back to log space
        # return combined.log()
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class CNN_PLUS(nn.Module):
    def __init__(self, input_channel, input_w, input_h, output_size, hidden_size):
        super(CNN_PLUS, self).__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(  # 输入的图片 (1,28,28)
                in_channels=input_channel,
                out_channels=hidden_size*2,  # 经过一个卷积层之后(64,28,28)
                kernel_size=5,
                stride=1,  # res_w = (m_w - kernel_size + 2*padding)/stride + 1
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 经过池化层处理，维度为（64，14，14）
            nn.Conv2d(  # 输入为（64，14，14）
                in_channels=hidden_size*2,
                out_channels=hidden_size*4,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # 输出为（128，7，7）
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 输出为（128，7, 7）
            nn.Flatten(),  # 平铺成128*7*7
            nn.Linear(in_features=128*7*7, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=output_size),
            nn.Softmax(1)  # 在第二个维度做softmax（第一个维度是batch_size）
        )

    def forward(self, x):
        out = self.branch(x)
        return out


class CNN(nn.Module):
    def __init__(self, input_channel, input_w, input_h, output_size, hidden_size):
        super(CNN, self).__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(  # 输入的图片 (1,28,28)
                in_channels=input_channel,
                out_channels=hidden_size,  # 经过一个卷积层之后(32,28,28)
                kernel_size=5,
                stride=1,  # res_w = (m_w - kernel_size + 2*padding)/stride + 1
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 经过池化层处理，维度为（32，14，14）
            nn.Conv2d(  # 输入为（32，14，14）
                in_channels=hidden_size,
                out_channels=hidden_size*2,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # 输出为（64，7，7）
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 输出为（64，7, 7）
            nn.Flatten(),  # 平铺成64*7*7
            nn.Linear(in_features=64*7*7, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=output_size),
            nn.Softmax(1)  # 在第二个维度做softmax（第一个维度是batch_size）
        )

    def forward(self, x):
        out = self.branch(x)
        return out


class GateCNN(nn.Module):
    def __init__(self, input_channel, input_w, input_h, output_size, hidden_size):
        super(GateCNN, self).__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(  # 输入的图片 (1,28,28)
                in_channels=input_channel,
                out_channels=hidden_size,  # 经过一个卷积层之后(32,28,28)
                kernel_size=5,
                stride=1,  # res_w = (m_w - kernel_size + 2*padding)/stride + 1
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 经过池化层处理，维度为（32，14，14）
            nn.Conv2d(  # 输入为（32，14，14）
                in_channels=hidden_size,
                out_channels=hidden_size*2,
                kernel_size=5,
                stride=1,
                padding=2
            ),  # 输出为（64，7，7）
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 输出为（64，7, 7）
            nn.Flatten(),  # 平铺成64*7*7
            nn.Linear(in_features=64*7*7, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=128, out_features=output_size),
            nn.Softmax(1)  # 在第二个维度做softmax（第一个维度是batch_size）
        )

    def forward(self, x):
        out = self.branch(x)
        return out


class GateCNN_NET(torch.nn.Module):
    def __init__(self, input_channel, input_w, input_h, output_size, hidden_size):
        super(GateCNN_NET, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=input_channel,
                                     out_channels=hidden_size,
                                     kernel_size=5,
                                     stride=1,
                                     padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(hidden_size, hidden_size*2, 5, stride=1, padding=2)
        self.fc1 = torch.nn.Linear(hidden_size * 2 * 8 * 8, 384)
        self.fc2 = torch.nn.Linear(384, 192)
        self.fc3 = torch.nn.Linear(192, output_size)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.soft = nn.Softmax(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # x:(64, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # x:(128, 8, 8)
        x = self.flatten(x)  # x:(128*8*8)
        x = F.relu(self.fc1(x))  # x:(384)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # x:(192)
        x = self.fc3(x)  # x:(outputsize)
        x = self.soft(x)
        return x


class CNN_NET(torch.nn.Module):
    def __init__(self, input_channel, input_w, input_h, output_size, hidden_size):
        super(CNN_NET, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=input_channel,
                                     out_channels=hidden_size,
                                     kernel_size=5,
                                     stride=1,
                                     padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(hidden_size, hidden_size * 2, 5, stride=1, padding=2)
        self.fc1 = torch.nn.Linear(hidden_size * 2 * 8 * 8, 384)
        self.fc2 = torch.nn.Linear(384, 192)
        self.fc3 = torch.nn.Linear(192, output_size)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.soft = nn.Softmax(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # x:(64, 16, 16) 16384
        x = self.pool(F.relu(self.conv2(x)))  # x:(128, 8, 8) 8192
        x = self.flatten(x)  # x:(128*8*8)
        x = F.relu(self.fc1(x))  # x:(384)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # x:(192)
        x = self.fc3(x)  # x:(outputsize)
        x = self.soft(x)
        return x


class CNN_NET_PLUS(torch.nn.Module):
    def __init__(self, input_channel, input_w, input_h, output_size, hidden_size):
        super(CNN_NET_PLUS, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=input_channel,
                                     out_channels=hidden_size*2,
                                     kernel_size=5,
                                     stride=1,
                                     padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(hidden_size*2, hidden_size * 4, 5, stride=1, padding=2)
        self.fc1 = torch.nn.Linear(hidden_size * 4 * 8 * 8, 384)
        self.fc2 = torch.nn.Linear(384, 192)
        self.fc3 = torch.nn.Linear(192, output_size)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.soft = nn.Softmax(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # x:(64, 16, 16)
        x = self.pool(F.relu(self.conv2(x)))  # x:(128, 8, 8)
        x = self.flatten(x)  # x:(128*8*8)
        x = F.relu(self.fc1(x))  # x:(384)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))  # x:(192)
        x = self.fc3(x)  # x:(outputsize)
        x = self.soft(x)
        return x

class MoE(nn.Module):
    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_channel, input_w, input_h, output_size, num_experts, hidden_size, dataset, noisy_gating=True, k=4, explore_step=10000,
                 random_gate=True, sparse_gating=False, n_main_experts=1):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_channel * input_w * input_h
        self.input_channel = input_channel
        self.input_w = input_w
        self.input_h = input_h
        self.hidden_size = hidden_size
        self.k = k
        self.dataset = dataset

        self.random_gate = random_gate
        self.sparse_gating = sparse_gating
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.explore_step = explore_step
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step

        # instantiate experts
        if self.dataset == 'mnist':
            self.experts = nn.ModuleList(
                [CNN(self.input_channel, self.input_w, self.input_h, self.output_size, self.hidden_size) for i in range(self.num_experts)])
            self.w_gate = GateCNN(self.input_channel, self.input_w, self.input_h, self.num_experts, self.hidden_size)
            self.w_noise = GateCNN(self.input_channel, self.input_w, self.input_h, self.num_experts, self.hidden_size)
        if self.dataset == 'cifar10':
            # CNN
            # Resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
            # self.experts = nn.ModuleList(
            #     [CNN_NET(self.input_channel, self.input_w, self.input_h, self.output_size, self.hidden_size) for i in range(self.num_experts)]
            # )
            # self.w_gate = GateCNN_NET(self.input_channel, self.input_w, self.input_h, self.num_experts,
            #                           self.hidden_size)
            # self.w_noise = GateCNN_NET(self.input_channel, self.input_w, self.input_h, self.num_experts,
            #                            self.hidden_size)
            # Resnet
            self.experts = nn.ModuleList(
                [ResNet18(num_classes=self.output_size) for i in range(self.num_experts)]
            )
            self.w_gate = ResNet18(num_classes=self.num_experts)
            self.w_noise = ResNet18(num_classes=self.num_experts)
            # self.w_noise = GateCNN_NET(self.input_channel, self.input_w, self.input_h, self.num_experts, self.hidden_size)
            # self.expert_fix = ResNet18(num_classes=self.output_size)
            self.n_main_experts = n_main_experts
            self.expert_fix = nn.ModuleList([ResNet18(num_classes=self.output_size)] * self.n_main_experts)
            self.alpha = nn.Parameter(torch.tensor(0.9))

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert (self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the pretrain will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.w_gate(x)
        # assert not torch.isnan(clean_logits).any(), "Clean logits contain NaN values."
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # if not self.sparse_gating:
        #     if train:
        #         self.k = self.num_experts
        #     else:
        #         self.k = 2
        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]

        if np.random.rand() <= self.epsilon and self.random_gate and train:
            indices = torch.zeros_like(top_k_indices)
            for b in range(top_k_indices.size(0)):
                random_index = np.random.choice(range(self.num_experts), self.k, replace=False)
                for i in range(self.k):
                    indices[b][i] = random_index[i]
            top_k_indices = indices

        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-1):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model_bert.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        gates, load = self.noisy_top_k_gating(x, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        #
        loss = self.cv_squared(importance) + self.cv_squared(load)
        # loss *= loss_coef

        dispatcher = SparseDispatcher(self.num_experts, gates)

        expert_inputs = dispatcher.dispatch(x)
        expert_inputs = list(expert_inputs)  # Convert tuple to list
        expert_inputs_new = []
        for i, temp in enumerate(expert_inputs):
            temp = temp.view(-1, self.input_channel, self.input_w, self.input_h)
            expert_inputs_new.append(temp)
        expert_inputs_new = tuple(expert_inputs_new)
        # gates = dispatcher.expert_to_gates()
        expert_outputs = [self.experts[i](expert_inputs_new[i]) for i in range(self.num_experts)]
        if self.n_main_experts == 1:
            y = self.alpha * dispatcher.combine(expert_outputs) + (1 - self.alpha) * self.expert_fix[0](x)
        else:
            y = dispatcher.combine(expert_outputs)
            for i in range(self.n_main_experts):
                y += self.expert_fix[i](x) * 0.5
        return y, loss, gates, self.alpha
