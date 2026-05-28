"""
CATCH model with frequency-domain adaptive bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ts_benchmark.baselines.catch.layers.RevIN import RevIN
from ts_benchmark.baselines.catch.layers.adaptive_bottleneck import AdaptiveBottleNeck
from ts_benchmark.baselines.catch.layers.channel_mask import channel_mask_generator
from ts_benchmark.baselines.catch.layers.cross_channel_Transformer import Trans_C


class CATCHModel(nn.Module):
    def __init__(self, configs, **kwargs):
        super(CATCHModel, self).__init__()

        self.revin_layer = RevIN(
            configs.c_in,
            affine=configs.affine,
            subtract_last=configs.subtract_last,
        )

        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.seq_len = configs.seq_len
        self.horizon = self.seq_len
        self.patch_num = int((configs.seq_len - configs.patch_size) / configs.patch_stride + 1)
        self.norm = nn.LayerNorm(self.patch_size)
        self.use_adaptive_bottleneck = getattr(configs, "use_adaptive_bottleneck", True)
        self.baseline_compatible = (
            not self.use_adaptive_bottleneck and getattr(configs, "baseline_compatible", True)
        )

        self.re_attn = True
        self.mask_generator = channel_mask_generator(input_size=configs.patch_size, n_vars=configs.c_in)
        self.frequency_transformer = Trans_C(
            dim=configs.cf_dim,
            depth=configs.e_layers,
            heads=configs.n_heads,
            mlp_dim=configs.d_ff,
            dim_head=configs.head_dim,
            dropout=configs.dropout,
            patch_dim=configs.patch_size * 2,
            horizon=self.horizon * 2,
            d_model=configs.d_model * 2,
            regular_lambda=configs.regular_lambda,
            temperature=configs.temperature,
        )

        # The bottleneck is applied to raw complex frequency patches before CFM.
        self.dc_lambda = getattr(configs, "dc_lambda", 0.005)
        self.bn_loss_coef = getattr(configs, "bn_loss_coef", 1e-2)
        self.routing_level = getattr(configs, "routing_level", "patch")
        self.use_frequency_router_features = getattr(configs, "use_frequency_router_features", True)
        self.freq_pos_dim = getattr(configs, "freq_pos_dim", 8)
        self.latest_norm_input = None
        self.latest_bottleneck_stats = None

        if self.use_adaptive_bottleneck:
            bn_dims = getattr(configs, "bn_dims", [64, 128, 256, 512])
            k_experts = getattr(configs, "k_experts", 2)
            bottleneck_residual = getattr(configs, "bottleneck_residual", "none")

            actual_repr_dim = configs.c_in * configs.patch_size * 2
            router_dim = actual_repr_dim
            if self.use_frequency_router_features:
                router_dim += 2 + self.freq_pos_dim
                self.freq_pos_embedding = nn.Parameter(torch.zeros(1, self.patch_num, self.freq_pos_dim))
                nn.init.normal_(self.freq_pos_embedding, mean=0.0, std=0.02)

            self.adaptive_bottleneck = AdaptiveBottleNeck(
                seq_len=self.patch_num,
                seq_dim=configs.d_model * 2,
                repr_dim=actual_repr_dim,
                bn_dims=bn_dims,
                noisy_gating=True,
                k=k_experts,
                routing_level=self.routing_level,
                router_dim=router_dim,
                residual_mode=bottleneck_residual,
            )

        self.head_nf_f = configs.d_model * 2 * self.patch_num
        self.n_vars = configs.c_in
        self.individual = configs.individual
        self.head_f1 = Flatten_Head(
            self.individual,
            self.n_vars,
            self.head_nf_f,
            configs.seq_len,
            head_dropout=configs.head_dropout,
        )
        self.head_f2 = Flatten_Head(
            self.individual,
            self.n_vars,
            self.head_nf_f,
            configs.seq_len,
            head_dropout=configs.head_dropout,
        )

        self.ircom = nn.Linear(self.seq_len * 2, self.seq_len)
        self.rfftlayer = nn.Linear(self.seq_len * 2 - 2, self.seq_len)
        self.final = nn.Linear(self.seq_len * 2, self.seq_len)
        self.get_r = nn.Linear(configs.d_model * 2, configs.d_model * 2)
        self.get_i = nn.Linear(configs.d_model * 2, configs.d_model * 2)

    def forward(self, z):
        z = self.revin_layer(z, "norm")
        self.latest_norm_input = z

        z = z.permute(0, 2, 1)
        z = torch.fft.fft(z)
        z1 = z.real
        z2 = z.imag

        z1 = z1.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)
        z2 = z2.unfold(dimension=-1, size=self.patch_size, step=self.patch_stride)

        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        batch_size = z1.shape[0]
        patch_num = z1.shape[1]
        c_in = z1.shape[2]
        patch_size = z1.shape[3]

        z1 = torch.reshape(z1, (batch_size * patch_num, c_in, patch_size))
        z2 = torch.reshape(z2, (batch_size * patch_num, c_in, patch_size))
        patch_energy = torch.sqrt(z1.pow(2) + z2.pow(2) + 1e-8).mean(dim=(1, 2))
        patch_energy = patch_energy.reshape(batch_size, patch_num, 1)
        patch_energy = torch.log1p(patch_energy)
        patch_energy = (patch_energy - patch_energy.mean(dim=1, keepdim=True)) / (
            patch_energy.std(dim=1, keepdim=True, unbiased=False) + 1e-6
        )
        z_cat = torch.cat((z1, z2), -1)

        channel_mask = self.mask_generator(z_cat)

        bottleneck_loss = z_cat.new_tensor(0.0)
        if self.use_adaptive_bottleneck:
            z_bn_input = z_cat.reshape(batch_size, patch_num, -1)
            router_input = z_bn_input
            if self.use_frequency_router_features:
                freq_coord = torch.linspace(0, 1, patch_num, device=z.device, dtype=z_bn_input.dtype)
                freq_coord = freq_coord.view(1, patch_num, 1).expand(batch_size, -1, -1)
                freq_pos = self.freq_pos_embedding[:, :patch_num, :].expand(batch_size, -1, -1)
                router_input = torch.cat([z_bn_input, patch_energy, freq_coord, freq_pos], dim=-1)

            z_bn_output, bottleneck_loss, self.latest_bottleneck_stats = self.adaptive_bottleneck(
                router_input=router_input,
                repr=z_bn_input,
                loss_coef=self.bn_loss_coef,
                return_stats=True,
            )
            z_cat = z_bn_output.reshape(batch_size * patch_num, c_in, patch_size * 2)

        z, dcloss = self.frequency_transformer(z_cat, channel_mask)
        if dcloss is None:
            dcloss = z.new_tensor(0.0)

        z1 = self.get_r(z)
        z2 = self.get_i(z)

        z1 = torch.reshape(z1, (batch_size, patch_num, c_in, z1.shape[-1]))
        z2 = torch.reshape(z2, (batch_size, patch_num, c_in, z2.shape[-1]))

        z1 = z1.permute(0, 2, 1, 3)
        z2 = z2.permute(0, 2, 1, 3)

        z1 = self.head_f1(z1)
        z2 = self.head_f2(z2)

        complex_z = torch.complex(z1, z2)

        z = torch.fft.ifft(complex_z)
        zr = z.real
        zi = z.imag
        z = self.ircom(torch.cat((zr, zi), -1))

        z = z.permute(0, 2, 1)
        z = self.revin_layer(z, "denorm")

        total_auxiliary_loss = self.dc_lambda * dcloss + bottleneck_loss
        return z, complex_z.permute(0, 2, 1), total_auxiliary_loss


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, seq_len, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears1 = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears1.append(nn.Linear(nf, seq_len))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear1 = nn.Linear(nf, nf)
            self.linear2 = nn.Linear(nf, nf)
            self.linear3 = nn.Linear(nf, nf)
            self.linear4 = nn.Linear(nf, seq_len)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears1[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = F.relu(self.linear1(x)) + x
            x = F.relu(self.linear2(x)) + x
            x = F.relu(self.linear3(x)) + x
            x = self.linear4(x)

        return x
