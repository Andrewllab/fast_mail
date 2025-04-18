from __future__ import annotations

from typing import Sequence

import open3d as o3d
import open3d.core as o3c
import torch
from tensordict import NonTensorData
from torch_geometric.data import Data

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform
from utils.o3d import o3d_to_torch, torch_to_o3d
from utils.pyg import apply_mask


class DenoisePointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        nb_points: int,
        radius: float,
        pcd_keys: str | Sequence[str] = "pcd",
    ) -> None:

        self.nb_points = nb_points
        self.radius = radius

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

        # TODO: batching
        pos = data.pos
        assert pos is not None
        pos = torch_to_o3d(pos)
        pcd = o3d.t.geometry.PointCloud(pos)

        filtered_pcd, mask = pcd.remove_radius_outliers(
            nb_points=self.nb_points, search_radius=self.radius
        )

        # cannot convert a boolean tensor directly back to torch, so we convert
        # to uint8 and then back to bool
        mask = o3d_to_torch(mask.to(o3c.Dtype.UInt8)).to(torch.bool)

        data = apply_mask(data, mask)

        return data
