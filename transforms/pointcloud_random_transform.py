from __future__ import annotations

from typing import Sequence

from tensordict import TensorDict
from torch_geometric.data import Data

from environments.specs import DataSpecs
from transforms.pointmap_random_transform import BaseRandomlyTransform
from utils.math import transform_pointcloud


class RandomlyTransformPointCloudTrajectory(BaseRandomlyTransform):
    """Apply a random translation and rotation to the entire pointcloud
    trajectory during preprocessing, which is then constant throughout
    training.

    Args:
        specs (DataSpecs): The data specifications.
    """

    def __init__(
        self,
        specs: DataSpecs,
        max_translation: Sequence[float],
        max_angle: float | Sequence[float] | None = None,
        pcd_keys: str | Sequence[str] = "pcd",
    ):
        super().__init__(max_translation, max_angle)

        self._specs = specs
        if isinstance(pcd_keys, str):
            pcd_keys = [pcd_keys]
        else:
            pcd_keys = list(pcd_keys)
        self._pcd_keys = pcd_keys

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        device = tensordict.device or "cpu"

        transform = self.random_pose(device)

        for key in self._pcd_keys:
            data: Data = tensordict["obs", key]
            data.pos = transform_pointcloud(data.pos, transform)

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Do nothing if not called during preprocessing
        return tensordict
