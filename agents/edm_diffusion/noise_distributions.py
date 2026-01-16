from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import Tensor, device, dtype

from agents.edm_diffusion.gc_sampling import get_sigmas_exponential

ShapeType = tuple[int, ...]
DeviceType = device | str | int


class NoiseDistributionType(Protocol):
    def __call__(self, shape: ShapeType, device: DeviceType) -> Tensor: ...


def rand_log_normal(
    shape: ShapeType,
    loc: float,
    scale: float,
    device: DeviceType = "cpu",
    dtype: dtype = torch.float32,
) -> Tensor:
    """Draws samples from a lognormal distribution."""
    loc = math.log(loc)
    return (torch.randn(shape, device=device, dtype=dtype) * scale + loc).exp()


def rand_log_logistic(
    shape: ShapeType,
    loc: float,
    scale: float,
    min_value: float | None = None,
    max_value: float | None = None,
    device: DeviceType = "cpu",
    dtype: dtype = torch.float32,
) -> Tensor:
    """Draws samples from an optionally truncated log-logistic distribution."""
    loc = math.log(loc)

    if min_value is not None:
        min_val = torch.as_tensor(min_value, device=device, dtype=torch.float64)
        min_cdf = min_val.log().sub(loc).div(scale).sigmoid()
    else:
        min_cdf = 0.0
    if max_value is not None:
        max_val = torch.as_tensor(max_value, device=device, dtype=torch.float64)
        max_cdf = max_val.log().sub(loc).div(scale).sigmoid()
    else:
        max_cdf = 1.0

    u = (
        torch.rand(shape, device=device, dtype=torch.float64) * (max_cdf - min_cdf)
        + min_cdf
    )
    return u.logit().mul(scale).add(loc).exp().to(dtype)


def rand_log_uniform(
    shape: ShapeType,
    min_value: float,
    max_value: float,
    device: DeviceType = "cpu",
    dtype: dtype = torch.float32,
) -> Tensor:
    """Draws samples from an log-uniform distribution."""
    min_value = math.log(min_value)
    max_value = math.log(max_value)
    return (
        torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value)
        + min_value
    ).exp()


def rand_uniform(
    shape: ShapeType,
    min_value: float,
    max_value: float,
    device: DeviceType = "cpu",
    dtype: dtype = torch.float32,
) -> Tensor:
    """Draws samples from a uniform distribution."""
    return (
        torch.rand(shape, device=device, dtype=dtype) * (max_value - min_value)
        + min_value
    )


def rand_v_diffusion(
    shape: ShapeType,
    sigma_data: float,
    sigma_min: float,
    sigma_max: float,
    device: DeviceType = "cpu",
    dtype: dtype = torch.float32,
) -> Tensor:
    """Draws samples from a truncated v-diffusion training timestep distribution."""
    min_cdf = math.atan(sigma_min / sigma_data) * 2 / math.pi
    max_cdf = math.atan(sigma_max / sigma_data) * 2 / math.pi
    u = torch.rand(shape, device=device, dtype=dtype) * (max_cdf - min_cdf) + min_cdf
    return torch.tan(u * math.pi / 2) * sigma_data


def rand_split_log_normal(
    shape: ShapeType,
    loc: float,
    scale_1: float,
    scale_2: float,
    device: DeviceType = "cpu",
    dtype: dtype = torch.float32,
) -> Tensor:
    """Draws samples from a split lognormal distribution."""
    n = torch.randn(shape, device=device, dtype=dtype).abs()
    u = torch.rand(shape, device=device, dtype=dtype)
    n_left = n * -scale_1 + loc
    n_right = n * scale_2 + loc
    ratio = scale_1 / (scale_1 + scale_2)
    return torch.where(u < ratio, n_left, n_right).exp()


def rand_discrete(
    shape: ShapeType,
    sigma_min: float,
    sigma_max: float,
    n_total_steps: int,
    device: DeviceType = "cpu",
    dtype: dtype = torch.float32,
):
    """Draws samples from the given discrete values."""
    sigmas = get_sigmas_exponential(
        n=n_total_steps, sigma_min=sigma_min, sigma_max=sigma_max, device=device
    )

    indices = torch.randint(0, len(sigmas), shape, device=device)
    samples = torch.index_select(sigmas, 0, indices).to(dtype)
    return samples
