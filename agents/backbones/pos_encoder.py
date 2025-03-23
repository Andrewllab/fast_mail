from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


class SinusoidalPosEncoder(nn.Module):
    """Positional encoding according to Attention Is All You Need.
    https://arxiv.org/abs/1706.03762
    """

    frequencies: Tensor

    def __init__(self, dim: int):
        super().__init__()
        half_dim = dim // 2

        # exponents from 0.0 to 1.0
        exponents = torch.arange(half_dim) / (half_dim - 1)

        # equivalent to this, but avoids division
        # frequencies = 1 / torch.pow(10000, exponents)
        frequencies = torch.exp(-math.log(10000) * exponents)

        # unsqueeze to allow broadcasting input position with all frequencies
        self.register_buffer("frequencies", frequencies.unsqueeze(dim=0))

    def forward(self, x: Tensor) -> Tensor:
        arg = x.unsqueeze(dim=-1) * self.frequencies
        return torch.cat((arg.sin(), arg.cos()), dim=-1)
