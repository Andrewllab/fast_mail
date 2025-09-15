from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor

from environments.specs import CameraSpec, DataSpecs, PointMapStream
from transforms.base_transform import Transform


class CropPointMap(Transform, nn.Module):
    """Crop the point map to a specified bounding box.

    This transform can also be used on batches of point clouds.

    Args:
        specs (DataSpecs): The data specifications.
        x_range (tuple[float, float]): The x-axis range for cropping.
        y_range (tuple[float, float]): The y-axis range for cropping.
        z_range (tuple[float, float]): The z-axis range for cropping.
    """

    min_bound: Tensor
    max_bound: Tensor

    def __init__(
        self,
        specs: DataSpecs,
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
    ):
        super().__init__()

        x_range = x_range or (-torch.inf, torch.inf)
        y_range = y_range or (-torch.inf, torch.inf)
        z_range = z_range or (-torch.inf, torch.inf)

        # register buffers so they get moved to the correct device with the module
        self.register_buffer(
            "min_bound", torch.tensor([x_range[0], y_range[0], z_range[0]])
        )
        self.register_buffer(
            "max_bound", torch.tensor([x_range[1], y_range[1], z_range[1]])
        )

        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def forward(self, tensordict: TensorDict) -> TensorDict:
        for key, spec in self._specs.obs.items():
            if not isinstance(spec, CameraSpec):
                continue
            for name, stream in spec.streams.items():
                if not isinstance(stream, PointMapStream):
                    continue

                pointmap: Tensor = tensordict["obs", key, name]
                min_bound = self.min_bound.to(pointmap.device)
                max_bound = self.max_bound.to(pointmap.device)
                
                pos = pointmap[..., :3]  # if it has colors, ignore them for cropping
                mask = ((pos >= min_bound) & (pos <= max_bound)).all(dim=-1)
                # set out-of-bounds points (and their colors) to zero
                pointmap[~mask] = 0.0

        return tensordict
