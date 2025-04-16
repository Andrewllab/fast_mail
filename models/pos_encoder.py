from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTokenPosEncoder(nn.Module):
    """Positional encoding according to Attention Is All You Need.
    https://arxiv.org/abs/1706.03762
    """

    frequencies: Tensor

    def __init__(self, embed_dim: int, temperature: float = 10000.0):
        super().__init__()

        # we only want half as many exponents as n_dim, because we have sin and
        # cos components
        half_dim = embed_dim // 2

        # exponents from 0.0 to 1.0
        exponents = torch.arange(half_dim) / (half_dim - 1)

        # equivalent to this, but avoids division
        # frequencies = 1 / torch.pow(10000, exponents)
        frequencies = torch.exp(-math.log(temperature) * exponents)

        # unsqueeze to allow broadcasting input position with all frequencies
        self.register_buffer("frequencies", frequencies.unsqueeze(dim=0))

    def forward(self, x: Tensor) -> Tensor:
        arg = x.unsqueeze(dim=-1) * self.frequencies
        return torch.cat((arg.sin(), arg.cos()), dim=-1)


class SinusoidalCartesianPosEncoder(nn.Module):

    frequencies: Tensor

    def __init__(
        self,
        embed_dim: int,
        cartesian_dim: int = 3,
        temperature: float = 10000.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        # divide the embedding dimension among the cartesian dimensions we have
        # to encode, ensuring that the result is even
        n_dim = embed_dim // cartesian_dim // 2 * 2

        # because of rounding down, this is the number of dimensions that our
        # encoding actually uses
        self.non_empty_embed_dim = n_dim * cartesian_dim

        # we only want half as many exponents as n_dim, because we have sin and
        # cos components
        half_dim = n_dim // 2

        # exponents from 0.0 to 1.0
        exponents = torch.arange(half_dim) / (half_dim - 1)

        # equivalent to this, but avoids division
        # frequencies = 1 / torch.pow(10000, exponents)
        frequencies = torch.exp(-math.log(temperature) * exponents)

        # unsqueeze to allow broadcasting input position with all frequencies
        self.register_buffer("frequencies", frequencies.unsqueeze(dim=0))

        # adjust the scale so that coordinates of 1 are the maxima of sin/cos
        self.scale = scale * 2 * torch.pi

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        pos_emb = torch.zeros(
            *pos.shape[:-1], self.embed_dim, dtype=pos.dtype, device=pos.device
        )

        pos = pos * self.scale

        # multiply each coordinate by each frequency using broadcasting, then flatten
        arg = pos.unsqueeze(-1) * self.frequencies
        arg = torch.flatten(arg, start_dim=-2)

        # fill in alternating sin/cos components
        end = self.non_empty_embed_dim
        pos_emb[..., 0:end:2] = arg.sin()
        pos_emb[..., 1 : end + 1 : 2] = arg.cos()

        return pos_emb
