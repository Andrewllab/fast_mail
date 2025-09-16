from __future__ import annotations

from typing import Sequence

import torch
from tensordict import TensorDict
from torch import Tensor

from environments.specs import CameraSpec, DataSpecs, PointMapStream
from transforms.base_transform import Transform
from utils.math import (
    make_pose,
    quaternion_to_matrix,
    random_orientation,
    sample_uniform,
    transform_pointmap,
)


class RandomlyTransformPointMapTrajectory(Transform):
    """Apply a random translation and rotation to the entire point map
    trajectory during preprocessing, which is then constant throughout
    training.

    Args:
        specs (DataSpecs): The data specifications.
    """

    def __init__(
        self,
        specs: DataSpecs,
        max_translation: Sequence[float],
    ):
        super().__init__()

        self.max_translation = torch.tensor(max_translation)
        if self.max_translation.shape != (3,):
            raise ValueError("max_translation must be a sequence of 3 floats")

        self._specs = specs

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
        transform = make_pose(translation, rot)

        for key, spec in self._specs.obs.items():
            if not isinstance(spec, CameraSpec):
                continue
            for name, stream in spec.streams.items():
                if not isinstance(stream, PointMapStream):
                    continue

                pointmap: Tensor = tensordict["obs", key, name]
                pos = pointmap[..., :3]  # if it has colors, do not modify them
                pointmap[..., :3] = transform_pointmap(pos, transform)

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Do nothing if not called during preprocessing
        return tensordict
