from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

from environments.specs import CameraSpec, DataSpecs, DepthStream, PointMapStream
from transforms.base_transform import NormalizingTransform

log = logging.getLogger(__name__)


class PointMapMaxMinNormalize(NormalizingTransform, nn.Module):
    quantiles: torch.Tensor | None
    max_points: torch.Tensor
    min_points: torch.Tensor

    def __init__(
        self,
        specs: DataSpecs,
        cutoff_depth: float | None = None,
        quantile: float = 1.0,
        scale: float = 1.0,
        clamp: bool = False,
        do_reverse: bool = False,
    ):
        super().__init__()

        if not 0.5 < quantile <= 1.0:
            raise ValueError(
                f"Quantile must be in the range (0.5, 1.0]. Received {quantile}"
            )

        if quantile < 1.0:
            self.register_buffer("quantiles", torch.tensor([quantile, 1 - quantile]))
        else:
            self.quantiles = None

        if scale <= 0.0:
            raise ValueError(f"Scale must be positive. Received {scale}")

        self.cutoff_depth = cutoff_depth
        self.scale = scale
        self.clamp = clamp
        self.do_reverse = do_reverse
        self._specs = specs

        self.streams = {
            (key, name): (spec, stream)
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            for name, stream in spec.streams.items()
            if isinstance(stream, PointMapStream)
        }
        if not self.streams:
            raise ValueError("No pointmap streams found in specs")

        channels = [stream.channels for (spec, stream) in self.streams.values()]
        if not all(c == channels[0] for c in channels):
            raise ValueError(
                f"All camera streams must have the same number of channels, but got {channels}"
            )
        channels = channels[0]

        if channels != 3:
            raise ValueError(
                f"Currently only support 3-channel pointmaps, got {channels}"
            )

        self.register_buffer("max_points", torch.full((channels,), float("-inf")))
        self.register_buffer("min_points", torch.full((channels,), float("inf")))

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict):
        for (key, name), (cam_spec, pm_stream) in self.streams.items():
            pointmap: torch.Tensor = tensordict["obs", key, name]

            # (..., H, W, C) -> (N, C)
            points = pointmap.flatten(0, -2)
            assert points.dim() == 2
            assert points.shape[-1] == 3

            if self.cutoff_depth is not None:
                depth_name = next(
                    name
                    for name, stream in cam_spec.streams.items()
                    if isinstance(stream, DepthStream)
                )
                depth = tensordict["obs", key, depth_name]
                mask = depth.flatten() < self.cutoff_depth
                if not mask.all():
                    points = points[mask]

            # Collect max and min values
            if self.quantiles is None:
                max_points = points.max(dim=0).values
                min_points = points.min(dim=0).values
            else:
                max_points, min_points = quantile(points, self.quantiles, dim=0)

            # accumulate max and min values
            self.max_points = torch.maximum(self.max_points, max_points)
            self.min_points = torch.minimum(self.min_points, min_points)

        log.debug(f"Updated max coordinate bounds: {self.max_points}")
        log.debug(f"Updated min coordinate bounds: {self.min_points}")

        return tensordict

    def forward(self, tensordict):
        if torch.isinf(self.max_points).any() or torch.isinf(self.min_points).any():
            raise ValueError(
                "Found inf in max_points or min_points. Make sure to call call_trajectory on a representative dataset before using the transform."
            )

        if (self.max_points <= self.min_points).any():
            raise ValueError(
                "max_points must be greater than min_points for all channels."
            )

        for key, name in self.streams.keys():
            pointmap: torch.Tensor = tensordict["obs", key, name]

            # Scale pointmap from [min_points, max_points] to [-scale, scale] on each axis
            pointmap.sub_(self.min_points).div_(self.max_points - self.min_points)
            pointmap.mul_(2 * self.scale).sub_(self.scale)

            if self.clamp:
                pointmap.clamp_(min=-1.0, max=1.0)

        return tensordict

    def reverse(self, tensordict):
        if not self.do_reverse:
            return tensordict

        for key, name in self.streams.keys():
            pointmap: torch.Tensor = tensordict["obs", key, name]

            # Unscale pointmap from [-scale, scale] to [min_points, max_points] on each axis
            pointmap.add_(self.scale).div(2 * self.scale)
            pointmap.mul_(self.max_points - self.min_points).add_(self.min_points)

        return tensordict


def quantile(
    tensor: Tensor, q: float | Tensor, dim: int | None = None, keepdim: bool = False
):
    """
    Computes the quantile of the input tensor along the specified dimension.

    Unfortunately torch.quantile is very slow and has a limit of 16 million elements.
    See: https://github.com/pytorch/pytorch/issues/64947
    Modified from: https://github.com/pytorch/pytorch/issues/64947#issuecomment-2810054982

    Parameters:
    tensor (torch.Tensor): The input tensor.
    q (float): The quantile to compute, should be a float between 0 and 1.
    dim (int): The dimension to reduce. If None, the tensor is flattened.
    keepdim (bool): Whether to keep the reduced dimension in the output.
    Returns:
    torch.Tensor: The quantile value(s) along the specified dimension.
    """
    q = torch.as_tensor(q, dtype=tensor.dtype, device=tensor.device)

    assert (0 <= q).all() and (q <= 1).all()

    if dim is None:
        tensor = tensor.flatten()
        dim = 0
    elif dim > 0:
        raise NotImplementedError

    sorted_tensor, _ = torch.sort(tensor, dim=dim)
    num_elements = sorted_tensor.size(dim)
    index = q * (num_elements - 1)
    lower_index = index.floor()
    upper_index = (lower_index + 1).clamp(max=num_elements - 1)
    # lower_value = sorted_tensor.select(dim, lower_index.long())
    # upper_value = sorted_tensor.select(dim, upper_index.long())
    lower_value = sorted_tensor[lower_index.long()]
    upper_value = sorted_tensor[upper_index.long()]

    # linear interpolation
    weight = index - lower_index
    weight = weight[(...,) + (None,) * (tensor.ndim - weight.ndim)]
    quantile_value = (1 - weight) * lower_value + weight * upper_value

    return quantile_value.unsqueeze(dim) if keepdim else quantile_value
