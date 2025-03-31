from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

    from environments.specs import DataSpecs

log = logging.getLogger(__name__)


class Scaler(nn.Module):
    @property
    def lower_bound(self) -> Tensor: ...

    @property
    def upper_bound(self) -> Tensor: ...

    def normalize(self, y: Tensor) -> Tensor: ...

    def unnormalize(self, y: Tensor) -> Tensor: ...

    def clip_output(self, y: Tensor) -> Tensor: ...


class MinMaxScaler(Scaler):
    """Scales the data such that the max/min are mapped to +/-1."""

    y_min: Tensor
    y_max: Tensor
    y_range: Tensor

    def __init__(self, specs: DataSpecs):
        super().__init__()
        self.register_buffer("y_min", specs.action.a_min.clone())
        self.register_buffer("y_max", specs.action.a_max.clone())
        self.register_buffer("y_range", self.y_max - self.y_min)

        if (self.y_range == 0.0).any():
            log.warning(
                "Some dimensions of the action space have a range of zero. This will cause NaN values after action normalization."
            )

    @property
    def lower_bound(self) -> Tensor:
        return -1 * torch.ones_like(self.y_range)

    @property
    def upper_bound(self) -> Tensor:
        return torch.ones_like(self.y_range)

    @torch.no_grad()
    def normalize(self, y: Tensor) -> Tensor:
        return (y - self.y_min) / self.y_range * 2 - 1

    @torch.no_grad()
    def unnormalize(self, y: Tensor) -> Tensor:
        return (y + 1) / 2 * self.y_range + self.y_min

    @torch.no_grad()
    def clip_output(self, y: Tensor) -> Tensor:
        return torch.clamp(y, self.lower_bound * 1.1, self.upper_bound * 1.1)


class NormalizingScaler(Scaler):
    y_mean: Tensor
    y_std: Tensor
    y_min: Tensor
    y_max: Tensor

    def __init__(self, specs: DataSpecs):
        super().__init__()
        self.register_buffer("y_mean", specs.action.a_mean.clone())
        self.register_buffer("y_std", specs.action.a_std.clone())
        self.register_buffer("y_min", specs.action.a_min.clone())
        self.register_buffer("y_max", specs.action.a_max.clone())

        if (self.y_std == 0.0).any():
            log.warning(
                "Some dimensions of the action space have a variance of zero. This will cause very large values after action normalization."
            )

    @property
    def lower_bound(self) -> Tensor:
        return (self.y_min - self.y_mean) / (self.y_std + 1e-12)

    @property
    def upper_bound(self) -> Tensor:
        return (self.y_max - self.y_mean) / (self.y_std + 1e-12)

    @torch.no_grad()
    def normalize(self, y: Tensor) -> Tensor:
        return (y - self.y_mean) / (self.y_std + 1e-12)

    @torch.no_grad()
    def unnormalize(self, y: Tensor) -> Tensor:
        return y * (self.y_std + 1e-12) + self.y_mean

    @torch.no_grad()
    def clip_output(self, y: Tensor) -> Tensor:
        return torch.clamp(y, self.lower_bound * 1.1, self.upper_bound * 1.1)
