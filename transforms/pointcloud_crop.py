from __future__ import annotations

from typing import Sequence

import torch
from tensordict import NonTensorData
from torch_geometric.data import Data

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform
from utils.pyg import apply_mask


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
        x_range: tuple[float, float] | None = None,
        y_range: tuple[float, float] | None = None,
        z_range: tuple[float, float] | None = None,
        pcd_keys: str | Sequence[str] = "pcd",
    ):
        x_range = x_range or (-torch.inf, torch.inf)
        y_range = y_range or (-torch.inf, torch.inf)
        z_range = z_range or (-torch.inf, torch.inf)

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

    def _call_one(self, nt_data: NonTensorData) -> Data:
        data: Data = nt_data.data  # unpack NonTensorData wrapper around pyg Data object

        pos = data.pos
        assert pos is not None

        min_bound = torch.tensor(self.min_bound, device=pos.device)
        max_bound = torch.tensor(self.max_bound, device=pos.device)

        mask = ((pos >= min_bound) & (pos <= max_bound)).all(dim=-1)

        data = apply_mask(data, mask)

        return data
