from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SinusoidalTokenPosEncoder(nn.Module):
    """Positional encoding according to Attention Is All You Need.
    Compute a positional embedding to be added onto a token from the a token's
    position in the sequence. The embedding is sin/cos components with
    frequencies decreasing exponentially from 1 to 1/temperature.

    https://arxiv.org/abs/1706.03762
    """

    frequencies: Tensor

    def __init__(self, embed_dim: int, temperature: float = 10000.0):
        super().__init__()

        # we only want half as many frequencies as n_dim, because we have sin and
        # cos components
        n_frequencies = embed_dim // 2

        # exponents from 0.0 to 1.0
        exponents = torch.linspace(start=0, end=1, steps=n_frequencies)

        # frequencies decreasing exponentially
        # equivalent to the following, but avoids division
        # frequencies = 1 / torch.pow(10000, exponents)
        frequencies = torch.exp(-math.log(temperature) * exponents)

        self.register_buffer("frequencies", frequencies)

    def forward(self, x: Tensor) -> Tensor:
        arg = x.unsqueeze(dim=-1) * self.frequencies
        return torch.cat((arg.sin(), arg.cos()), dim=-1)


class PointGPTCartesianPosEncoder(nn.Module):
    """Positional encoding used by PointGPT.
    Computes a positional embedding to be added onto a token from some
    cartesian coordintes, presumably the center of the point patch. The
    embedding is sin/cos components with frequencies decreasing exponentially
    from 2*pi*scale to 2*pi*scale/temperature.

    Code: https://github.com/CGuangyan-BIT/PointGPT
    Paper: https://arxiv.org/abs/2305.11487
    """

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
        n_frequencies = n_dim // 2

        # exponents from 0.0 to 1.0
        exponents = torch.linspace(start=0, end=1, steps=n_frequencies)

        # frequencies decreasing exponentially
        # equivalent to the following, but avoids division
        # frequencies = 1 / torch.pow(10000, exponents)
        frequencies = torch.exp(-math.log(temperature) * exponents)

        # highest frequency is 2*pi*scale, lowest is 2*pi*scale/10000
        frequencies = frequencies * 2 * torch.pi * scale

        # unsqueeze to allow broadcasting input position with all frequencies
        self.register_buffer("frequencies", frequencies.unsqueeze(dim=0))

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        pos_emb = torch.zeros(
            *pos.shape[:-1], self.embed_dim, dtype=pos.dtype, device=pos.device
        )

        # multiply each coordinate by each frequency using broadcasting, then flatten
        arg = pos.unsqueeze(-1) * self.frequencies
        arg = torch.flatten(arg, start_dim=-2)

        # fill in alternating sin/cos components
        end = self.non_empty_embed_dim
        pos_emb[..., 0:end:2] = arg.sin()
        pos_emb[..., 1 : end + 1 : 2] = arg.cos()

        return pos_emb


class PointMAECartesianPosEncoder(nn.Module):
    """Positional encoding used by PointMAE (Point Masked Autoencoder).
    Computes a positional embedding to be added onto a token from some
    cartesian coordintes, presumably the center of the point patch. The
    embedding is learned.

    Code: https://github.com/Pang-Yatian/Point-MAE
    Paper: https://arxiv.org/abs/2203.06604
    """

    def __init__(
        self, embed_dim: int, cartesian_dim: int = 3, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(cartesian_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, pos: Tensor) -> Tensor:
        return self.model(pos)


class NerfPositionProjection(nn.Module):
    """Positional encoding used by NeRF.
    Projects the coordinates of 3D point into a higher dimensional space, which
    is more suitable as input to a neural network than raw xyz, as these values
    change very slowly. The embedding is sin/cos components with frequencies
    increasing exponentially from 2*pi*scale/max_wavelength to
    2*pi*scale/min_wavelength. The final embedding will have
    2*n_wavelengths*cartesian_dim dimensions.
    """

    frequencies: Tensor

    def __init__(
        self,
        n_wavelengths: int,
        max_wavelength: float,
        min_wavelength: float,
        cartesian_dim: int = 3,
        scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.cartesian_dim = cartesian_dim

        # frequencies increase exponentially from 1/max_wavelength to 1/min_wavelength
        exponents = torch.linspace(
            start=math.log(max_wavelength),
            end=math.log(min_wavelength),
            steps=n_wavelengths,
        )
        frequencies = torch.exp(-exponents)

        frequencies = frequencies * 2 * torch.pi * scale

        # unsqueeze to allow broadcasting input position with all frequencies
        self.register_buffer("frequencies", frequencies.unsqueeze(dim=0))

    def forward(self, pos: torch.Tensor) -> torch.Tensor:

        # multiply each coordinate by each frequency using broadcasting, then flatten
        arg = pos.unsqueeze(-1) * self.frequencies
        arg = torch.flatten(arg, start_dim=-2)

        # take sin and code of each argument, and concatenate
        embedding = torch.cat((arg.sin(), arg.cos()), dim=-1)

        return embedding

    @property
    def in_features(self) -> int:
        """Returns the input size of the model."""
        return self.cartesian_dim

    @property
    def out_features(self) -> int:
        """Retuns the output size of the model."""
        return self.cartesian_dim * 2 * self.frequencies.shape[-1]
