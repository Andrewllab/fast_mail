from __future__ import annotations

from typing import Sequence

import torch
from tensordict import NonTensorData
from torch_geometric.data import Data
from torch_geometric.nn import radius

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform
from utils.pyg import apply_mask


class RemoveRadiusOutliers(Transform):
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
        pos, batch = data.pos, data.batch
        batch_size = getattr(data, "batch_size", None)
        assert pos is not None

        src_idxs, target_idxs = radius(
            x=pos,
            y=pos,
            r=self.radius,
            batch_x=batch,
            batch_y=batch,
            max_num_neighbors=self.nb_points,
            batch_size=batch_size,
        )

        neighbor_counts = torch.bincount(src_idxs, minlength=pos.size(0))
        mask = neighbor_counts >= self.nb_points

        data = apply_mask(data, mask)

        return data
