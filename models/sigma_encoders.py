from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from models.pos_encoder import SinusoidalSequencePosEncoder


class DDPMSigmaEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.sigma_emb = nn.Sequential(
            SinusoidalSequencePosEncoder(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.Mish(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, sigma: Tensor) -> Tensor:
        return self.sigma_emb(sigma)
