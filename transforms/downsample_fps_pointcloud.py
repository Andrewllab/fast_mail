from typing import Sequence

from torch_geometric.data import Data
from torch_geometric.nn import fps

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform
from utils.pyg import apply_index


class FpsSamplePointCloud(Transform):
    """Downsample a point cloud using farthest point sampling (FPS).

    This transform can also be used on batches of point clouds.

    Args:
        specs (DataSpecs): The data specifications.
        target_size (int | None): The target number of points to sample. If None, ratio must be provided.
        ratio (float | None): The ratio of points to sample. If None, target_size must be provided.
        pcd_keys (str | Sequence[str]): The keys of the point cloud data to downsample.
    """

    def __init__(
        self,
        specs: DataSpecs,
        target_size: int | None = None,
        ratio: float | None = None,
        pcd_keys: str | Sequence[str] = "pcd",
    ) -> None:
        if target_size is not None:
            raise NotImplementedError

        if target_size is None and ratio is None:
            raise ValueError("Either target_size or ratio must be provided.")
        elif target_size is not None and ratio is not None:
            raise ValueError("Only one of target_size or ratio may be provided.")

        self.target_size = target_size
        self.ratio = ratio

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
        num_points = data.num_nodes
        assert num_points is not None
        assert data.pos is not None
        ratio = self.ratio
        assert ratio is not None

        batch_size = data.batch_size if hasattr(data, "batch_size") else None
        # https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.pool.fps.html
        fps_idx = fps(data.pos, data.batch, ratio=ratio, batch_size=batch_size)

        data = apply_index(data, fps_idx)

        return data
