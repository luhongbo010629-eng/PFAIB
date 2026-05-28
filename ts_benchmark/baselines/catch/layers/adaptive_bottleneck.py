"""
Adaptive bottleneck module for CATCH frequency-domain patches.
"""

import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class BottleNeck(nn.Module):
    def __init__(self, seq_len, repr_dim, bn_dim, residual_mode="none"):
        super(BottleNeck, self).__init__()
        self.seq_len = seq_len
        self.repr_dim = repr_dim
        self.residual_mode = residual_mode
        self.net = nn.Sequential(
            nn.Linear(repr_dim, bn_dim),
            nn.GELU(),
            nn.Linear(bn_dim, bn_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bn_dim, repr_dim),
        )
        if residual_mode == "scaled":
            self.res_scale = nn.Parameter(torch.tensor(0.1))
        elif residual_mode != "none":
            raise ValueError(f"Unsupported bottleneck residual_mode: {residual_mode}")

    def forward(self, repr):
        bottlenecked = self.net(repr)
        if self.residual_mode == "scaled":
            return bottlenecked + self.res_scale * repr
        return bottlenecked


class AdaptiveBottleNeck(nn.Module):
    """
    Sparse MoE bottleneck.

    routing_level="patch" routes each frequency patch independently and stores
    latest_gates as [batch, patch_num, num_experts].

    routing_level="window" keeps the original window-level behavior and stores
    latest_gates as [batch, num_experts].
    """

    def __init__(
        self,
        seq_len,
        seq_dim,
        repr_dim,
        bn_dims=[32, 64, 128, 256],
        noisy_gating=True,
        k=2,
        routing_level="patch",
        router_dim=None,
        residual_mode="none",
    ):
        super(AdaptiveBottleNeck, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_bottlenecks = len(bn_dims)
        self.k = k
        self.seq_len = seq_len
        self.repr_dim = repr_dim
        self.seq_dim = seq_dim
        self.routing_level = routing_level
        self.router_dim = router_dim or repr_dim

        self.bottlenecks = nn.ModuleList(
            [BottleNeck(seq_len, repr_dim, bn_dim, residual_mode=residual_mode) for bn_dim in bn_dims]
        )

        if routing_level == "window":
            gate_input_dim = seq_len * self.router_dim
        elif routing_level == "patch":
            gate_input_dim = self.router_dim
        else:
            raise ValueError(f"Unsupported routing_level: {routing_level}")

        self.w_gate = nn.Parameter(torch.empty(gate_input_dim, self.num_bottlenecks))
        self.w_noise = nn.Parameter(torch.empty(gate_input_dim, self.num_bottlenecks))
        self.noise_bias = nn.Parameter(torch.full((self.num_bottlenecks,), -2.0))
        nn.init.xavier_uniform_(self.w_gate)
        nn.init.xavier_uniform_(self.w_noise, gain=0.01)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        self.latest_gates = None
        self.latest_balance = None

        assert self.k <= self.num_bottlenecks, f"k ({k}) must be <= num_bottlenecks ({self.num_bottlenecks})"

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return x.new_tensor(0.0)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)

        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        return torch.where(is_in, prob_if_in, prob_if_out)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate

        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise + self.noise_bias
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_bottlenecks), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        gates = torch.zeros_like(logits).scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_bottlenecks and train:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)

        return gates, load

    def forward(self, router_input, repr, loss_coef=1e-2, return_stats=False):
        batch_size, patch_num, _ = repr.shape

        if self.routing_level == "patch":
            router_flat = router_input.reshape(batch_size * patch_num, -1)
            repr_for_dispatch = repr.reshape(batch_size * patch_num, 1, self.repr_dim)
        else:
            router_flat = router_input.reshape(batch_size, -1)
            repr_for_dispatch = repr.reshape(batch_size, self.seq_len, self.repr_dim)

        gates, load = self.noisy_top_k_gating(router_flat, self.training)
        if self.routing_level == "patch":
            self.latest_gates = gates.detach().reshape(batch_size, patch_num, self.num_bottlenecks)
        else:
            self.latest_gates = gates.detach()

        importance = gates.sum(0)
        importance_cv = self.cv_squared(importance)
        load_cv = self.cv_squared(load)
        loss = (importance_cv + load_cv) * loss_coef
        stats = {
            "importance_cv": importance_cv.detach(),
            "load_cv": load_cv.detach(),
            "importance": importance.detach(),
            "load": load.detach(),
        }
        self.latest_balance = stats

        dispatcher = SparseDispatcher(self.num_bottlenecks, gates)
        expert_inputs = dispatcher.dispatch(repr_for_dispatch)

        expert_outputs = []
        for i in range(self.num_bottlenecks):
            if expert_inputs[i].size(0) > 0:
                expert_outputs.append(self.bottlenecks[i](expert_inputs[i]))
            else:
                expert_outputs.append(
                    torch.zeros(
                        0,
                        repr_for_dispatch.size(1),
                        self.repr_dim,
                        device=repr.device,
                        dtype=repr.dtype,
                    )
                )

        repr_combined = dispatcher.combine(expert_outputs)
        if self.routing_level == "patch":
            repr = repr_combined.reshape(batch_size, patch_num, self.repr_dim)
        else:
            repr = repr_combined.reshape(batch_size, self.seq_len, self.repr_dim)

        if return_stats:
            return repr, loss, stats
        return repr, loss


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts

        nonzero_indices = torch.nonzero(gates, as_tuple=False)
        _, sorted_by_expert = nonzero_indices[:, 1].sort(0)
        sorted_indices = nonzero_indices[sorted_by_expert]

        self._batch_index = sorted_indices[:, 0]
        self._expert_index = sorted_indices[:, 1].unsqueeze(1)
        self._part_sizes = (gates > 0).sum(0).tolist()

        gates_exp = gates[self._batch_index]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index]
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched * self._nonzero_gates.unsqueeze(-1)

        output_shape = (self._gates.size(0), expert_out[0].size(1), expert_out[0].size(2))
        combined = stitched.new_zeros(output_shape)
        return combined.index_add(0, self._batch_index, stitched)

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)
