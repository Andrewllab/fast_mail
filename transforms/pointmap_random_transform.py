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
    euler_to_matrix
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
        max_angle: float | Sequence[float] | None = None,
    ):
        self.max_translation = torch.tensor(max_translation)
        if self.max_translation.shape != (3,):
            raise ValueError("max_translation must be a sequence of 3 floats")

        self._specs = specs
        
        if isinstance(max_angle, (float, int)):
            self.max_angle = torch.tensor([max_angle, max_angle, max_angle])
        elif isinstance(max_angle, Sequence):
            self.max_angle = torch.tensor(max_angle)
            if self.max_angle.shape != (3,):
                raise ValueError("max_angle must be a float or a sequence of 3 floats")
        elif max_angle is None:
            self.max_angle = None

        if self.max_angle is not None and not ((0 <= self.max_angle) & (self.max_angle <= 180)).all().item():
            raise ValueError("max_angle must be in [0, 180]")
        
        # Convert to radians
        if self.max_angle is not None:
            self.max_angle = self.max_angle * torch.pi / 180.0

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        device = tensordict.device or "cpu"

        if self.max_angle is None:
            quat = random_orientation(num=1, device=device)
            rot = quaternion_to_matrix(quat)
        else:
            random_euler = sample_uniform(
                    lower=-self.max_angle,
                    upper=self.max_angle,
                    size=(1, 3),
                    device=device,
                )
            rot = euler_to_matrix(random_euler, convention="XYZ")

        translation = sample_uniform(
            lower=-self.max_translation,
            upper=self.max_translation,
            size=(1, 3),
            device="cpu",
        ).to(device)

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