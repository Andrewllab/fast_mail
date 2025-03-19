import math

import einops
import numpy as np
import torch
import torch.nn as nn


def return_time_sigma_embedding_model(embedding_type, time_embed_dim, device):
    """
    Method returns an embedding model given the chosen type
    """
    if embedding_type == "GaussianFourier":
        return GaussianFourierEmbedding(time_embed_dim, device)
    elif embedding_type == "Sinusoidal":
        return SinusoidalPosEmbedding(time_embed_dim, device)
    elif embedding_type == "FourierFeatures":
        return FourierFeatures(time_embed_dim, device)
    else:
        raise ValueError("Embedding not avaiable, please chose an existing one!")


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierFeatures(nn.Module):
    def __init__(self, time_embed_dim, device, in_features=1, std=1.0):
        super().__init__()
        self.device = device
        assert time_embed_dim % 2 == 0
        self.register_buffer(
            "weight", torch.randn([time_embed_dim // 2, in_features]) * std
        )

    def forward(self, input):
        if len(input.shape) == 1:
            input = einops.rearrange(input, "b -> b 1")
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1).to(self.device)


class GaussianFourierEmbedding(nn.Module):

    def __init__(self, time_embed_dim, device):
        super().__init__()
        self.t_dim = time_embed_dim
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=time_embed_dim),
            nn.Linear(time_embed_dim, 2 * time_embed_dim),
            nn.Mish(),
            nn.Linear(2 * time_embed_dim, time_embed_dim),
        ).to(device)

    def forward(self, t):
        return self.embed(t)


class SinusoidalPosEmbedding(nn.Module):

    def __init__(self, time_embed_dim, device):
        super().__init__()
        self.device = device
        self.embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.Mish(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        ).to(self.device)

    def forward(self, t):
        return self.embed(t)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class InputEncoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.emb = nn.Linear(self.input_dim, self.latent_dim)

    def forward(self, x):
        return self.emb(x)


class TEncoder(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.emb = nn.Linear(self.input_dim, self.latent_dim)

    def forward(self, x):
        return self.emb(x)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]
