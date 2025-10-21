from __future__ import annotations

import logging
from typing import Sequence

import torch
from tensordict import NonTensorData, TensorDict
from torch_geometric.data import Data

from environments.datamodule import EmptyPointCloudError
from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform
from utils.pyg import apply_mask

log = logging.getLogger(__name__)


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
        warn_min_points: int = 100,
    ):
        x_range = x_range or (-torch.inf, torch.inf)
        y_range = y_range or (-torch.inf, torch.inf)
        z_range = z_range or (-torch.inf, torch.inf)
        self.warn_min_points = warn_min_points
        self.error_min_points = False

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

        num_points = data.ptr[1:] - data.ptr[:-1]
        if (num_points <= self.warn_min_points).any():
            log.warning(
                f"Some point clouds are empty after cropping. The number of points per batch element is: {num_points}"
            )
            if self.error_min_points:
                raise EmptyPointCloudError("Aborting due to empty point clouds.")

        return data

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        error_min_points = self.error_min_points
        self.error_min_points = True
        tensordict = self(tensordict)
        self.error_min_points = error_min_points
        return tensordict
