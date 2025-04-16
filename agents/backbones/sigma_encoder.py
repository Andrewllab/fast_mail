from __future__ import annotations

from typing import TYPE_CHECKING

import einops
import torch.nn as nn

from models.pos_encoder import SinusoidalTokenPosEncoder

if TYPE_CHECKING:
    from torch import Tensor


class BESO_SigmaEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.sigma_emb = nn.Sequential(
            SinusoidalTokenPosEncoder(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, sigma: Tensor) -> Tensor:
        sigma = sigma.log() / 4
        sigma = einops.rearrange(sigma, "b -> b 1")
        sigma_emb = self.sigma_emb(sigma)
        return sigma_emb


class DDPM_SigmaEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.sigma_emb = nn.Sequential(
            SinusoidalTokenPosEncoder(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, sigma: Tensor) -> Tensor:
        sigma = einops.rearrange(sigma, "b -> b 1")
        return self.sigma_emb(sigma)
