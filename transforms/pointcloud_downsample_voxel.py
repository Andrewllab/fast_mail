import re
from typing import Sequence

import torch
from tensordict import NonTensorData
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils import one_hot, scatter

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform


class GridSamplePointCloud(Transform):
    """Clusters points into fixed-sized voxels
    (functional name: :obj:`grid_sampling`).
    Each cluster returned is a new point based on the mean of all points
    inside the given cluster.

    Args:
        specs (DataSpecs): The data specifications.
        size (float or [float] or Tensor): Size of a voxel (in each dimension).
        start (float or [float] or Tensor, optional): Start coordinates of the
            grid (in each dimension). If set to :obj:`None`, will be set to the
            minimum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
        end (float or [float] or Tensor, optional): End coordinates of the grid
            (in each dimension). If set to :obj:`None`, will be set to the
            maximum coordinates found in :obj:`data.pos`.
            (default: :obj:`None`)
        pcd_keys (str | Sequence[str]): The keys of the point cloud data to downsample.

    Modified from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/grid_sampling.html
    """

    def __init__(
        self,
        specs: DataSpecs,
        size: float,
        start: float | Tensor | None = None,
        end: float | Tensor | None = None,
        pcd_keys: str | Sequence[str] = "pcd",
    ) -> None:

        self.size = size
        self.start = start
        self.end = end

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

        assert data.pos is not None
        c = voxel_grid(data.pos, self.size, data.batch, self.start, self.end)
        c, perm = consecutive_cluster(c)

        for key, item in data.items():
            if bool(re.search("edge", key)):
                raise ValueError(
                    f"'{self.__class__.__name__}' does not "
                    f"support coarsening of edges"
                )

            if torch.is_tensor(item) and item.size(0) == num_nodes:
                if key == "y":
                    item = scatter(one_hot(item), c, dim=0, reduce="sum")
                    data[key] = item.argmax(dim=-1)
                elif key == "batch":
                    data[key] = item[perm]
                else:
                    data[key] = scatter(item, c, dim=0, reduce="mean")

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"
