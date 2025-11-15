from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


class LearnableTokenEncoder(nn.Module):
    """Embed Layer with learnable parameters.
    For 2D positional encoding.
    Published with the Vision Transformer (ViT) architecture.
    Used in Multi-View-Transformer (MVT) like RVT, RVT2.

    Code: https://github.com/s-chh/2D-Positional-Encoding-Vision-Transformer/blob/main/positional_encodings/pos_embed_learn.py
    Paper: https://arxiv.org/abs/2010.11929
    """

    def __init__(self, n_channels, embed_dim, image_size, patch_size, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            n_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )  # Patch Encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, (image_size // patch_size) ** 2, embed_dim),
            requires_grad=True,
        )  # Learnable Positional Embedding
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim), requires_grad=True
        )  # Classification Token
        self.dropout = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.conv1.weight, mean=0.0, std=0.02)
        nn.init.constant_(self.conv1.bias, 0)
        nn.init.trunc_normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.conv1(
            x
        )  # B, C, IH, IW     -> B, E, IH/P, IW/P                Split image into the patches and embed patches
        x = x.reshape(
            [B, x.shape[1], -1]
        )  # B, E, IH/P, IW/P -> B, E, (IH/P*IW/P) -> B, E, N    Flattening the patches
        x = x.permute(
            0, 2, 1
        )  # B, E, N          -> B, N, E                         Rearrange to put sequence dimension in the middle
        x = (
            x + self.pos_embedding
        )  # B, N, E          -> B, N, E                         Add positional embedding
        # x = torch.cat(
        #     (torch.repeat_interleave(self.cls_token, B, 0), x), dim=1
        # )  # B, N, E          -> B, (N+1), E       -> B, S, E    Add classification token at the start of every sequence
        x = self.dropout(x)
        return x


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
        cartesian_dim: int,
        embed_dim: int,
        temperature: float = 10000.0,
        scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.cartesian_dim = cartesian_dim

        # divide the embedding dimension among the cartesian dimensions we have
        # to encode, dividing by 2 because we have sin and cos components
        n_frequencies = embed_dim // cartesian_dim // 2

        # because of rounding down, this is the number of dimensions that our
        # encoding actually uses
        self.non_empty_embed_dim = n_frequencies * 2 * cartesian_dim

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

    @property
    def in_features(self) -> int:
        """Returns the input size of the model."""
        return self.cartesian_dim

    @property
    def out_features(self) -> int:
        """Retuns the output size of the model."""
        return self.embed_dim


class PointMAECartesianPosEncoder(nn.Module):
    """Positional encoding used by PointMAE (Point Masked Autoencoder).
    Computes a positional embedding to be added onto a token from some
    cartesian coordintes, presumably the center of the point patch. The
    embedding is learned.

    Code: https://github.com/Pang-Yatian/Point-MAE
    Paper: https://arxiv.org/abs/2203.06604
    """

    def __init__(
        self, cartesian_dim: int, embed_dim: int, hidden_dim: int = 128
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(cartesian_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, pos: Tensor) -> Tensor:
        return self.model(pos)

    @property
    def in_features(self) -> int:
        """Returns the input size of the model."""
        return self.model[0].in_features

    @property
    def out_features(self) -> int:
        """Retuns the output size of the model."""
        return self.model[-1].out_features


class FourierFeatures(nn.Module):
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
        input_dim: int,
        # embed_dim: int | None = None,
        n_wavelengths: int | None = None,
        max_wavelength: float | None = None,
        min_wavelength: float | None = None,
        interval: float | None = None,
        learnable: bool = False,
        mix_dimensions: bool = True,
        cat_input_to_out: bool = False,
        components: Literal["sin", "sincos"] = "sincos",
        scale: float = 1.0,
    ) -> None:
        super().__init__()

        # if embed_dim is None:
        #     assert n_wavelengths is not None

        #     embed_dim = feature_dim = input_dim * (
        #         # sin and cos components for each frequency
        #         # add 1 for the original coordinates if we are concatenating them
        #         2 * n_wavelengths
        #         + (1 if cat_input_to_out else 0)
        #     )
        #     padding_dim = 0
        # else:
        #     assert n_wavelengths is None

        #     n_wavelengths = embed_dim // (input_dim * 2) - (
        #         1 if cat_input_to_out else 0
        #     )
        #     feature_dim = input_dim * (
        #         2 * n_wavelengths + (1 if cat_input_to_out else 0)
        #     )
        #     padding_dim = embed_dim - feature_dim

        if interval is None:
            assert n_wavelengths is not None
            assert max_wavelength is not None
            assert min_wavelength is not None

            # frequencies increase exponentially from 1/max_wavelength to 1/min_wavelength
            log_wavelengths = torch.linspace(
                start=math.log(max_wavelength),
                end=math.log(min_wavelength),
                steps=n_wavelengths,
            )

            if len(log_wavelengths) >= 2:
                interval = torch.exp(log_wavelengths[1] - log_wavelengths[0]).item()
                log.debug(
                    f"Using fourier features with {n_wavelengths} wavelengths from {max_wavelength} to {min_wavelength} with an interval of {interval:.3f}"
                )

        elif min_wavelength is None:
            assert n_wavelengths is not None
            assert max_wavelength is not None
            assert interval is not None

            log_wavelengths = torch.arange(n_wavelengths) * math.log(interval)
            log_wavelengths += math.log(max_wavelength)

            log.debug(
                f"Using fourier features with {n_wavelengths} wavelengths from {max_wavelength} to {torch.exp(log_wavelengths[-1])} with an interval of {interval}"
            )

        elif max_wavelength is None:
            assert n_wavelengths is not None
            assert min_wavelength is not None
            assert interval is not None

            log_wavelengths = -torch.arange(n_wavelengths - 1, -1, -1) * math.log(
                interval
            )
            log_wavelengths += math.log(min_wavelength)

            log.debug(
                f"Using fourier features with {n_wavelengths} wavelengths from {torch.exp(log_wavelengths[0])} to {min_wavelength} with an interval of {interval}"
            )

        else:
            assert n_wavelengths is None
            assert max_wavelength is not None
            assert min_wavelength is not None
            assert interval is not None

            log_wavelengths = torch.arange(
                math.log(max_wavelength), math.log(min_wavelength), math.log(interval)
            )
            n_wavelengths = len(log_wavelengths)

            log.debug(
                f"Using fourier features with {n_wavelengths} wavelengths from {max_wavelength} to {min_wavelength} with an interval of {interval}"
            )

        frequencies = torch.exp(-log_wavelengths)
        frequencies = frequencies * 2 * torch.pi * scale
        self.register_buffer("frequencies", frequencies)

        # add 1 for the original coordinates if we are concatenating them
        feature_dim = input_dim * (2 * n_wavelengths + (1 if cat_input_to_out else 0))
        padding_dim = 0
        embed_dim = feature_dim

        self.input_dim = input_dim
        self.cat_input_to_out = cat_input_to_out
        self.embed_dim = embed_dim
        self.padding_dim = padding_dim

        if learnable:
            # sin and cos components for each frequency
            fourier_feature_dim = 2 * n_wavelengths
            if mix_dimensions:
                fourier_feature_dim *= input_dim
            self.linear = nn.Linear(fourier_feature_dim, fourier_feature_dim)
        self.learnable = learnable
        self.mix_dimensions = mix_dimensions

    def forward(self, pos: torch.Tensor) -> torch.Tensor:

        # multiply each coordinate by each frequency using broadcasting
        # pos: (..., input_dim)
        # arg: (..., input_dim, n_frequencies)
        arg = pos.unsqueeze(dim=-1) * self.frequencies

        # take sin and code of each argument
        features = (arg.sin(), arg.cos())

        if self.learnable:
            # concatenate sin(ωx) with cos(ωx) for each dimension
            features = torch.cat(features, dim=-1)

            if self.mix_dimensions:
                # flatten the last two dimensions to mix features of different coordinates
                features = features.flatten(start_dim=-2)

            features = torch.sin(self.linear(features))

            if self.mix_dimensions:
                # unflatten back to (..., input_dim, 2 * n_frequencies)
                features = features.unflatten(
                    dim=-1, sizes=(pos.shape[-1], 2 * len(self.frequencies))
                )

            features = (features,)

        if self.cat_input_to_out:
            # concatenate the original coordinates with the sin/cos components
            features = (pos.unsqueeze(dim=-1), *features)

        if self.padding_dim > 0:
            padding_shape = arg.shape[:-1] + (self.padding_dim,)
            features = (*features, arg.new_zeros(padding_shape))

        if len(features) > 1:
            features = torch.cat(features, dim=-1)
        else:
            features = features[0]

        # flatten the last two dimensions to mix features of different coordinates
        # features -> (..., input_dim * D)
        features = features.flatten(start_dim=-2)

        return features

    @property
    def in_features(self) -> int:
        """Returns the input size of the model."""
        return self.input_dim

    @property
    def out_features(self) -> int:
        """Retuns the output size of the model."""
        return self.embed_dim

    def extra_repr(self):
        wavelengths = 2 * torch.pi / self.frequencies
        # remove the "tensor()" from the repr string
        wavelengths_s = repr(wavelengths)[7:-1]

        s = ", ".join(
            [
                f"in_features={self.in_features}",
                f"out_features={self.out_features}",
                f"wavelengths={wavelengths_s}",
                f"learnable={self.learnable}",
                f"cat_input_to_out={self.cat_input_to_out}",
            ]
        )

        return s
