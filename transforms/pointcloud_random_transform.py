from __future__ import annotations

from typing import Sequence

import torch
from tensordict import TensorDict
from torch_geometric.data import Data

from environments.specs import DataSpecs
from transforms.base_transform import Transform
from utils.math import (
    make_pose,
    quaternion_to_matrix,
    random_orientation,
    sample_uniform,
    transform_pointcloud,
)


class RandomlyTransformPointCloudTrajectory(Transform):
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
        pcd_keys: str | Sequence[str] = "pcd",
    ):
        self.max_translation = torch.tensor(max_translation)
        if self.max_translation.shape != (3,):
            raise ValueError("max_translation must be a sequence of 3 floats")

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

        quat = random_orientation(num=1, device=device)
        translation = sample_uniform(
            lower=-self.max_translation,
            upper=self.max_translation,
            size=(1, 3),
            device="cpu",
        ).to(device)

        rot = quaternion_to_matrix(quat)
        transform = make_pose(translation, rot).squeeze(0)

        for key in self._pcd_keys:
            data: Data = tensordict["obs", key]
            data.pos = transform_pointcloud(data.pos, transform)

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Do nothing if not called during preprocessing
        return tensordict
