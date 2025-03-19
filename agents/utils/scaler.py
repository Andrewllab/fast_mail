import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

log = logging.getLogger(__name__)


class Scaler(nn.Module):
    def __init__(self, y_data: Tensor):
        raise NotImplementedError

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

    def __init__(self, y_data: Tensor):
        self.register_buffer("y_min", y_data.min(0))
        self.register_buffer("y_max", y_data.max(0))
        self.register_buffer("y_range", self.y_max - self.y_min)

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

    def __init__(self, y_data: Tensor):
        self.register_buffer("y_mean", y_data.mean(0))
        self.register_buffer("y_std", y_data.std(0))
        self.register_buffer("y_min", y_data.min(0))
        self.register_buffer("y_max", y_data.max(0))

        log.info(f"Action lower bounds across dataset:\n{self.lower_bound}")
        log.info(f"Action upper bounds across dataset:\n{self.upper_bound}")

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
