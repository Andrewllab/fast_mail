from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Data

from transforms.base_transform import KeyMapping, Transform


if TYPE_CHECKING:
    from torch import Tensor

    from environments.specs import DataSpecs


class CropPointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        x_range: tuple[float, float] = (-1.0, 1.0),
        y_range: tuple[float, float] = (-1.0, 1.0),
        z_range: tuple[float, float] = (-1.0, 1.0),
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        
        self.specs = specs

        self._key_mappings = [
            KeyMapping(
                in_keys=[("obs", "pcd")],
                out_keys=[("obs", "pcd")]
                )]

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return self._key_mappings

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, pc: Data) -> Data:

        min_bound = torch.tensor([self.x_range[0], self.y_range[0], self.z_range[0]])
        max_bound = torch.tensor([self.x_range[1], self.y_range[1], self.z_range[1]])

        mask = ((pc.pos >= [min_bound]) & (pc.pos <= max_bound)).all(dim=1)

        data = {"pos": pc.pos[mask]}

        if hasattr(pc, "x") and pc.x is not None and pc.x.size(0) == pc.pos.size(0):
            data["x"] = pc.x[mask]
        
        return Data(**data)