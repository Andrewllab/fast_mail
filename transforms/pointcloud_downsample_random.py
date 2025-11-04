import logging
import math
from typing import Sequence

import numpy as np
import torch
from tensordict import NonTensorData
from torch_geometric.data import Data

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform
from utils.pyg import apply_index, update_batch_metadata

log = logging.getLogger(__name__)


class RandomSamplePointCloud(Transform):
    r"""Randomly samples a fixed number of points from a point cloud.

    Args:
        specs (DataSpecs): The data specifications.
        n_points (int): The number of points to sample.
        replace (bool, optional): If set to :obj:`False`, samples points
            without replacement. (default: :obj:`False`)
        allow_duplicates (bool, optional): In case :obj:`replace` is
            :obj`False` and :obj:`n_points` is greater than the number of points,
            this option determines whether to add duplicated nodes to the
            output points or not.
            In case :obj:`allow_duplicates` is :obj:`False`, the number of
            output points might be smaller than :obj:`n_points`.
            In case :obj:`allow_duplicates` is :obj:`True`, the number of
            duplicated points are kept to a minimum. (default: :obj:`False`)
        pcd_keys (str | Sequence[str]): The keys of the point cloud data to downsample.

    Modified from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/fixed_points.html#FixedPoints
    """

    def __init__(
        self,
        specs: DataSpecs,
        n_points: int,
        replace: bool = False,
        allow_duplicates: bool = False,
        pcd_keys: str | Sequence[str] = "pcd",
    ) -> None:

        self.n_points = n_points
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

    def _call_one(self, nt_data: NonTensorData) -> Data:
        data: Data = nt_data.data  # unpack NonTensorData wrapper around pyg Data object

        num_nodes = data.num_nodes
        assert num_nodes is not None

        n_points = self.n_points
        if hasattr(data, "batch_size"):
            n_points *= data.batch_size

        if self.replace:
            choice = torch.from_numpy(
                np.random.choice(num_nodes, n_points, replace=True)
            ).long()
        elif self.allow_duplicates:
            choice = torch.cat(
                [
                    torch.randperm(num_nodes)
                    for _ in range(math.ceil(n_points / num_nodes))
                ],
                dim=0,
            )[:n_points]
        else:
            # automatically handles case where num_nodes < n_points
            choice = torch.randperm(num_nodes)[:n_points]

        data = apply_index(data, choice)
        data = update_batch_metadata(data)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_points={self.n_points})"
