from __future__ import annotations

from typing import Literal, Sequence

import torch
from tensordict import TensorDict
from torch import Tensor
from torch_geometric.data import Data

from environments.specs import (
    CameraSpec,
    DataSpecs,
    DepthStream,
    PointCloudSpec,
    PointMapStream,
)
from transforms.base_transform import Transform


class RandomTranslationalJitter(Transform):
    r"""Translates point positions by randomly sampled translation values
    from a given distribution.

    Args:
        sigma (float): Magnitude of distribution from which jitter is sampled.
            If `distribution_type` is `uniform`, jitter is sampled from
            :math:`\mathcal{U}(-\sigma, +\sigma)`. If the distribution type is
            normal, jitter is sampled from :math:`\mathcal{N}(0, \sigma)`. The
            same sigma value is used for each dimension.
        distribution ("uniform" or "normal"): Type of distribution to sample
            jitter from.
        random_sigma (bool): If `True`, the sigma value above is treated as a
            maximum magnitude and the sigma for each batch element is randomly
            sampled from :math:`\mathcal{U}(0, \sigma)`. This has the effect of
            ensuring that some batch elements have low jitter applied.
            (default: `True`)
        types (Sequence of "depth", "pointmap", or "pointcloud"): What data
            types to apply jitter to. By default, jitter is applied to all
            supported types that are present.
        clamp_depth_nonnegative (bool): If `True`, depth maps are clamped from
            below to be non-negative. (default: `False`)
        enable_in_eval (bool): If `True`, jitter is applied even during evaluation.
            (default: `False`)
    """

    def __init__(
        self,
        specs: DataSpecs,
        sigma: float,
        distribution: Literal["uniform", "normal"] = "uniform",
        random_sigma: bool = True,
        types: Sequence[Literal["depth", "pointmap", "pointcloud"]] | None = None,
        clamp_depth_nonnegative: bool = False,
        enable_in_eval: bool = False,
    ):
        if distribution not in ("uniform", "normal"):
            raise ValueError(
                f"Only uniform and normal distributions are supported, got {distribution}"
            )

        self.sigma = sigma
        self.distribution = distribution
        self.random_sigma = random_sigma
        self.types = (
            list(types) if types is not None else ["depth", "pointmap", "pointcloud"]
        )
        self.clamp_depth_nonnegative = clamp_depth_nonnegative
        self.enable_in_eval = enable_in_eval

        if "depth" in self.types:
            self.depth_streams = {
                (key, name): (spec, stream)
                for key, spec in specs.obs.items()
                if isinstance(spec, CameraSpec)
                for name, stream in spec.streams.items()
                if isinstance(stream, DepthStream)
            }
        else:
            self.depth_streams = {}

        if "pointmap" in self.types:
            self.pointmap_streams = {
                (key, name): (spec, stream)
                for key, spec in specs.obs.items()
                if isinstance(spec, CameraSpec)
                for name, stream in spec.streams.items()
                if isinstance(stream, PointMapStream)
            }
        else:
            self.pointmap_streams = {}

        if "pointcloud" in self.types:
            self.pcd_keys = {
                key: spec
                for key, spec in specs.obs.items()
                if isinstance(spec, PointCloudSpec)
            }
        else:
            self.pcd_keys = {}

        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if not self.training and not self.enable_in_eval:
            return tensordict

        for key, name in self.pointmap_streams.keys():
            pointmap: Tensor = tensordict["obs", key, name]

            B = pointmap.size(0)
            sigma = self._get_sigma(pointmap, B=B)

            noise = self._get_noise(pointmap).mul_(sigma)
            pointmap.add_(noise)

        for key, name in self.depth_streams.keys():
            depth: Tensor = tensordict["obs", key, name]

            B = depth.size(0)
            sigma = self._get_sigma(depth, B=B)

            noise = self._get_noise(depth).mul_(sigma)
            depth.add_(noise)

            if self.clamp_depth_nonnegative:
                # clamp depth to be non-negative
                depth.clamp_(min=0.0)

        for key in self.pcd_keys.keys():
            data: Data = tensordict["obs", key]

            pos = data.pos
            assert pos is not None

            if hasattr(data, "batch_size"):
                B = data.batch_size
                batch = data.batch
            else:
                B = 1
                batch = Ellipsis

            sigma = self._get_sigma(pos, B=B)

            if torch.is_tensor(sigma):
                # for random sigma, expand to assign a sigma to each point
                # if sigma is a scalar, this is unnecessary
                sigma = sigma[batch]

            noise = self._get_noise(pos).mul_(sigma)
            pos.add_(noise)

        return tensordict

    def _get_sigma(self, tensor: Tensor, B: int = 1) -> Tensor:
        if self.random_sigma:
            shape = (B,) + (1,) * (tensor.ndim - 1)
            # ensure that sigma is on device
            return tensor.new_empty(shape).uniform_(0, self.sigma)
        else:
            return self.sigma

    def _get_noise(self, tensor: Tensor) -> Tensor:
        if self.distribution == "uniform":
            # rescale from [0, 1] to [-1, 1]
            return torch.rand_like(tensor).mul_(2.0).sub_(1.0)

        if self.distribution == "normal":
            return torch.randn_like(tensor)

    def __repr__(self) -> str:
        args = [
            f"sigma={self.sigma}",
            f"random_sigma={self.random_sigma}",
            f"enable_in_eval={self.enable_in_eval}",
        ]
        if self.types is not None:
            args.append(f"types={self.types}")
        if self.clamp_depth_nonnegative:
            args.append(f"clamp_depth_nonnegative={self.clamp_depth_nonnegative}")
        s = ", ".join(args)
        return f"{self.__class__.__name__}({s})"


# BackCompat
def TranslationalJitter(specs, sigma):
    return RandomTranslationalJitter(
        specs, sigma=sigma, distribution="normal", random_sigma=False
    )


# BackCompat
def VariableTranslationalJitter(specs, max_sigma):
    return RandomTranslationalJitter(
        specs, sigma=max_sigma, distribution="normal", random_sigma=True
    )


# BackCompat
def JitterPointCloud(specs, max_sigma, pcd_keys="pcd"):
    assert pcd_keys == "pcd"

    return RandomTranslationalJitter(specs, sigma=max_sigma, distribution="uniform")
