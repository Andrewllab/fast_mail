from __future__ import annotations

import logging
import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)


class FourierFeaturesBase(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_wavelengths: int,
        axis_aligned: bool,
        embed_dim: int | None = None,
        spe: bool = False,
        spe_axis_aligned: bool | None = None,
        components: Literal["sincos", "sin"] = "sincos",
        cat_input_to_out: bool = False,
    ) -> None:
        super().__init__()

        if components not in ("sincos", "sin"):
            raise ValueError("components must be 'sincos' or 'sin'")

        n_components = 2 if components == "sincos" else 1

        # add 1 for the original coordinates if we are concatenating them
        feature_dim = input_dim * (
            n_components * n_wavelengths + (1 if cat_input_to_out else 0)
        )
        if embed_dim is not None:
            if feature_dim > embed_dim:
                raise ValueError(
                    f"embed_dim={embed_dim} is too small for the given configuration. "
                    f"Minimum required is {feature_dim}."
                )
            padding_dim = embed_dim - feature_dim
        else:
            padding_dim = 0
            embed_dim = feature_dim

        if spe:
            if not axis_aligned and spe_axis_aligned:
                log.warning(
                    "Axis-aligned SPE is not possible with non-axis-aligned frequencies. Setting spe_axis_aligned=False."
                )
                spe_axis_aligned = False

            if spe_axis_aligned is None:
                # match the axis_aligned setting by default
                spe_axis_aligned = axis_aligned

            spe_feature_dim = n_components * n_wavelengths
            if not spe_axis_aligned:
                spe_feature_dim *= input_dim

            self.spe_linear = nn.Linear(spe_feature_dim, spe_feature_dim)
            self.spe_axis_aligned = spe_axis_aligned

        else:
            self.spe_linear = None

        self.axis_aligned = axis_aligned
        self.components = components
        self.cat_input_to_out = cat_input_to_out

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.padding_dim = padding_dim

    def forward(self, frequencies: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:

        if self.axis_aligned:
            # multiply each coordinate by each frequency using broadcasting
            # pos: (..., input_dim, 1) * (n_frequencies)
            # arg: (..., input_dim, n_frequencies)
            arg = pos.unsqueeze(dim=-1) * frequencies

        else:
            # matrix multiply coordinates with frequency vectors
            # this implicitly flattens the Cartesian dimensions
            # pos: (..., input_dim) @ (input_dim, input_dim * n_frequencies)
            # arg: (..., input_dim * n_frequencies)
            arg = pos @ frequencies

        if self.components == "sincos":
            # concatenate sin(ωx) and cos(ωx) of each feature
            # if axis-aligned, each coordinate is kept separate
            # features: (..., input_dim, 2 * n_frequencies) or (..., input_dim * 2 * n_frequencies)
            features = torch.cat((arg.sin(), arg.cos()), dim=-1)
        else:  # components == "sin"
            # features: (..., input_dim, n_frequencies) or (..., input_dim * n_frequencies)
            features = arg.sin()

        if self.spe_linear is not None:
            if self.axis_aligned and not self.spe_axis_aligned:
                # flatten Cartesian dimensions for linear layer
                # features -> (..., n * input_dim * n_frequencies)
                features = features.flatten(start_dim=-2)

            # linear transform of the features (dimensionality not changed)
            features = self.spe_linear(features).sin()

            if self.axis_aligned and self.spe_axis_aligned:
                # flatten Cartesian dimensions to match the non-axis-aligned case
                # features -> (..., n * input_dim * n_frequencies)
                features = features.flatten(start_dim=-2)

        elif self.axis_aligned:
            # flatten Cartesian dimensions to match the non-axis-aligned case
            # features -> (..., n * input_dim * n_frequencies)
            features = features.flatten(start_dim=-2)

        # Cartesian dimensions are now flattened in all cases
        # features: (..., n * input_dim * n_frequencies)
        features = (features,)

        if self.cat_input_to_out:
            # concatenate the original coordinates with the sin/cos components
            features = (pos, *features)

        if self.padding_dim > 0:
            padding_shape = pos.shape[:-1] + (self.padding_dim,)
            features = (*features, pos.new_zeros(padding_shape))

        if len(features) > 1:
            features = torch.cat(features, dim=-1)
        else:
            features = features[0]

        return features

    @property
    def in_features(self) -> int:
        """Returns the input size of the model."""
        return self.input_dim

    @property
    def out_features(self) -> int:
        """Retuns the output size of the model."""
        return self.embed_dim

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split("\n")
        lines = extra_lines

        main_str = self._get_name() + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1:
                main_str += extra_lines[0]
            else:
                main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class FourierFeatures(FourierFeaturesBase):
    def __init__(
        self,
        input_dim: int,
        n_wavelengths: int | None = None,
        max_wavelength: float | None = None,
        min_wavelength: float | None = None,
        factor: float | None = None,
        embed_dim: int | None = None,
        learn: Literal["frequencies", "logfrequencies", False] = False,
        spe: bool = False,
        spe_axis_aligned: bool = True,
        components: Literal["sincos", "sin"] = "sincos",
        cat_input_to_out: bool = False,
    ) -> None:
        if components not in ("sincos", "sin"):
            raise ValueError("components must be 'sincos' or 'sin'")

        n_components = 2 if components == "sincos" else 1

        if embed_dim is not None:
            max_n_wavelengths = (
                embed_dim - (input_dim if cat_input_to_out else 0)
            ) // (input_dim * n_components)
        else:
            max_n_wavelengths = None

        # If only 2 of the 4 parameters are provided, but we are given an embed_dim,
        # use the maximum number of wavelengths that fit in the embed_dim
        args = (max_wavelength, min_wavelength, factor)
        if (
            n_wavelengths is None
            and max_n_wavelengths is not None
            and len([arg for arg in args if arg is not None]) == 2
        ):
            n_wavelengths = max_n_wavelengths
            log.info(
                f"Using maximum n_wavelengths={n_wavelengths} given embed_dim={embed_dim}"
            )

        log_frequencies = linspace_logfrequencies(
            n_wavelengths=n_wavelengths,
            max_wavelength=max_wavelength,
            min_wavelength=min_wavelength,
            factor=factor,
        )

        n_wavelengths = len(log_frequencies)
        min_wavelength = torch.exp(-log_frequencies[-1]).item()
        max_wavelength = torch.exp(-log_frequencies[0]).item()
        if len(log_frequencies) >= 2:
            factor = torch.exp(log_frequencies[1] - log_frequencies[0]).item()
        else:
            factor = float("nan")
        log.info(
            f"Using {n_wavelengths} Fourier wavelengths from {max_wavelength:.3f} to {min_wavelength:.3f} with an factor of {factor:.3f}"
        )

        super().__init__(
            input_dim=input_dim,
            n_wavelengths=n_wavelengths,
            axis_aligned=True,  # logspace frequencies are always axis-aligned
            embed_dim=embed_dim,
            spe=spe,
            spe_axis_aligned=spe_axis_aligned,
            components=components,
            cat_input_to_out=cat_input_to_out,
        )

        if learn not in ("frequencies", "logfrequencies", False):
            raise ValueError(
                "learn must be one of 'frequencies', 'logfrequencies', or False"
            )

        if learn == "logfrequencies":
            self.log_frequencies = nn.Parameter(log_frequencies, requires_grad=True)
        else:
            frequencies = torch.exp(log_frequencies)
            frequencies = 2 * torch.pi * frequencies

            self.frequencies = nn.Parameter(
                frequencies, requires_grad=learn == "frequencies"
            )

        self.learn = learn
        self.n_wavelengths = n_wavelengths
        self.min_wavelength = min_wavelength
        self.max_wavelength = max_wavelength
        self.factor = factor

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        if self.learn == "logfrequencies":
            frequencies = torch.exp(self.log_frequencies)
            frequencies = 2 * torch.pi * frequencies
        else:
            frequencies = self.frequencies

        return super().forward(frequencies, pos)

    def extra_repr(self):
        args = [
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
            f"wavelengths=logspace({self.max_wavelength:.2f}, {self.min_wavelength:.4f}, n={self.n_wavelengths}, factor={self.factor:.2f})",
        ]

        if self.learn:
            args.append(f"learn={self.learn}")
        if self.spe_linear is not None:
            args.append(f"SPE=True [axis-aligned={self.spe_axis_aligned}]")

        components = "sin/cos" if self.components == "sincos" else "sin"
        args.append(f"components={components}")
        args.append(f"cat_input_to_out={self.cat_input_to_out}")

        s = ",\n".join(args)

        return s


class RandomFourierFeatures(FourierFeaturesBase):
    def __init__(
        self,
        input_dim: int,
        sigma: float,
        embed_dim: int | None = None,
        n_wavelengths: int | None = None,
        axis_aligned: bool = False,
        learn: bool = False,
        spe: bool = False,
        spe_axis_aligned: bool | None = None,
        components: Literal["sincos", "sin"] = "sincos",
        cat_input_to_out: bool = False,
    ) -> None:
        if components not in ("sincos", "sin"):
            raise ValueError("components must be 'sincos' or 'sin'")

        n_components = 2 if components == "sincos" else 1

        if embed_dim is not None:
            max_n_wavelengths = (
                embed_dim - (input_dim if cat_input_to_out else 0)
            ) // (input_dim * n_components)
        else:
            max_n_wavelengths = None

        if n_wavelengths is None:
            if max_n_wavelengths is not None:
                n_wavelengths = max_n_wavelengths
                log.info(
                    f"Using maximum n_wavelengths={n_wavelengths} given embed_dim={embed_dim}"
                )
            else:
                raise ValueError("either n_wavelengths or embed_dim must be provided")

        super().__init__(
            input_dim=input_dim,
            n_wavelengths=n_wavelengths,
            axis_aligned=axis_aligned,
            embed_dim=embed_dim,
            spe=spe,
            spe_axis_aligned=spe_axis_aligned,
            components=components,
            cat_input_to_out=cat_input_to_out,
        )

        if axis_aligned:
            # each frequency is a scalar, and is applied to each input dimension
            # separately
            shape = (n_wavelengths,)
        else:
            # each frequency is a random vector in R^input_dim
            # sample input_dim times as many frequencies so that we have the
            # same embed dim as for axis-aligned case
            shape = (input_dim, input_dim * n_wavelengths)

        # random Fourier frequencies sampled from N(0, sigma^2)
        frequencies = torch.empty(shape).normal_(mean=0.0, std=sigma)
        frequencies = 2 * torch.pi * frequencies
        self.frequencies = nn.Parameter(frequencies, requires_grad=learn)

        self.learn = learn
        self.sigma = sigma

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        return super().forward(self.frequencies, pos)

    def extra_repr(self):
        wavelengths = torch.linalg.norm(2 * torch.pi / self.frequencies, dim=0)

        args = [
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
            f"wavelengths=N(0, {self.sigma:.1f}^2) [max/min=({wavelengths.max():.3f}, {wavelengths.min():.3f})]",
            f"axis_aligned={self.axis_aligned}",
        ]

        if self.learn:
            args.append(f"learn={self.learn}")
        if self.spe_linear is not None:
            args.append(f"SPE=True [axis-aligned={self.spe_axis_aligned}]")

        components = "sin/cos" if self.components == "sincos" else "sin"
        args.append(f"components={components}")
        args.append(f"cat_input_to_out={self.cat_input_to_out}")

        s = ", ".join(args)

        return s


def linspace_logfrequencies(
    n_wavelengths: int | None = None,
    max_wavelength: float | None = None,
    min_wavelength: float | None = None,
    factor: float | None = None,
) -> Tensor:
    if n_wavelengths is not None and n_wavelengths < 2:
        raise ValueError("n_wavelengths must be at least 2 if provided.")

    if factor is None:
        if any(x is None for x in (n_wavelengths, max_wavelength, min_wavelength)):
            raise ValueError(
                "Must provide exactly 3 of the following parameters: n_wavelengths, max_wavelength, min_wavelength, factor."
            )
        assert n_wavelengths is not None
        assert max_wavelength is not None
        assert min_wavelength is not None

        # frequencies increase exponentially from 1/max_wavelength to 1/min_wavelength
        log_wavelengths = torch.linspace(
            start=math.log(max_wavelength),
            end=math.log(min_wavelength),
            steps=n_wavelengths,
        )

    elif min_wavelength is None:
        if any(x is None for x in (n_wavelengths, max_wavelength, factor)):
            raise ValueError(
                "Must provide exactly 3 of the following parameters: n_wavelengths, max_wavelength, min_wavelength, factor."
            )
        assert n_wavelengths is not None
        assert max_wavelength is not None
        assert factor is not None

        # wavelengths decrease exponentially from max_wavelength by factor
        log_wavelengths = -torch.arange(n_wavelengths) * math.log(factor)
        log_wavelengths += math.log(max_wavelength)

    elif max_wavelength is None:
        if any(x is None for x in (n_wavelengths, min_wavelength, factor)):
            raise ValueError(
                "Must provide exactly 3 of the following parameters: n_wavelengths, max_wavelength, min_wavelength, factor."
            )
        assert n_wavelengths is not None
        assert min_wavelength is not None
        assert factor is not None

        log_wavelengths = torch.arange(n_wavelengths - 1, -1, -1) * math.log(factor)
        log_wavelengths += math.log(min_wavelength)

    elif n_wavelengths is None:
        if any(x is None for x in (max_wavelength, min_wavelength, factor)):
            raise ValueError(
                "Must provide exactly 3 of the following parameters: n_wavelengths, max_wavelength, min_wavelength, factor."
            )
        assert max_wavelength is not None
        assert min_wavelength is not None
        assert factor is not None

        log_wavelengths = torch.arange(
            math.log(max_wavelength), math.log(min_wavelength), -math.log(factor)
        )
        n_wavelengths = len(log_wavelengths)

    else:
        raise ValueError(
            "Must provide exactly 3 of the following parameters: n_wavelengths, max_wavelength, min_wavelength, factor."
        )

    # frequencies are 1 / wavelengths
    return -log_wavelengths
