from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from transforms.base_transform import KeyMapping, Transform
from utils.pyg import apply_mask

if TYPE_CHECKING:
    from torch_geometric.data import Data

    from environments.specs import DataSpecs


class CropPointCloud(Transform):
    """Crop the point cloud to a specified bounding box.

    This transform can also be used on batches of point clouds.

    Args:
        specs (DataSpecs): The data specifications.
        x_range (tuple[float, float]): The x-axis range for cropping.
        y_range (tuple[float, float]): The y-axis range for cropping.
        z_range (tuple[float, float]): The z-axis range for cropping.
        pcd_keys (str | Sequence[str]): The keys of the point cloud data to crop.
    """

    def __init__(
        self,
        specs: DataSpecs,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
        pcd_keys: str | Sequence[str] = "pcd",
    ):
        self.min_bound = [x_range[0], y_range[0], z_range[0]]
        self.max_bound = [x_range[1], y_range[1], z_range[1]]

        self._specs = specs
        if isinstance(pcd_keys, str):
            pcd_keys = [pcd_keys]
        else:
            pcd_keys = list(pcd_keys)
        self._pcd_keys = pcd_keys

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(in_keys=[("obs", key)], out_keys=[("obs", key)])
            for key in self._pcd_keys
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, data: Data) -> Data:
        pos = data.pos
        assert pos is not None

        min_bound = torch.tensor(self.min_bound, device=pos.device)
        max_bound = torch.tensor(self.max_bound, device=pos.device)

        mask = ((pos >= [min_bound]) & (pos <= max_bound)).all(dim=-2)

        data = apply_mask(data, mask)

        return data
