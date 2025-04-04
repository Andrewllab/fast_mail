import logging
import math
from typing import Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform
from utils.pyg import apply_index

log = logging.getLogger(__name__)


class RandomSamplePointCloud(Transform):
    """Randomly sample a fixed number of points from a point cloud.

    This transform can only be used on individual point clouds.

    Modified from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/fixed_points.html
    """

    def __init__(
        self,
        specs: DataSpecs,
        num: int,
        replace: bool = True,
        allow_duplicates: bool = False,
        pcd_keys: str | Sequence[str] = "pcd",
    ) -> None:
        self.num = num
        self.replace = replace
        self.allow_duplicates = allow_duplicates

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
        num_nodes = data.num_nodes
        assert num_nodes is not None

        if self.replace:
            choice = torch.from_numpy(
                np.random.choice(num_nodes, self.num, replace=True)
            ).long()
        elif not self.allow_duplicates:
            if self.num >= num_nodes:
                log.debug(
                    f"Skipping random sampling of {self.num} points from point cloud with {num_nodes}."
                )
                return data
            choice = torch.randperm(num_nodes)[: self.num]
        else:
            choice = torch.cat(
                [
                    torch.randperm(num_nodes)
                    for _ in range(math.ceil(self.num / num_nodes))
                ],
                dim=0,
            )[: self.num]

        data = apply_index(data, choice)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num={self.num})"
