from __future__ import annotations

from typing import Sequence

import torch
from tensordict import TensorDict
from torch import Tensor

from environments.specs import CameraSpec, DataSpecs, PointMapStream
from transforms.base_transform import Transform
from utils.math import (
    euler_to_matrix,
    make_pose,
    quaternion_to_matrix,
    random_orientation,
    sample_uniform,
    transform_pointmap,
)


class BaseRandomlyTransform(Transform):
    def __init__(
        self,
        max_translation: Sequence[float],
        max_degrees: float | Sequence[float] | None = None,
    ):
        self.max_translation = torch.tensor(max_translation)
        if self.max_translation.shape != (3,):
            raise ValueError("max_translation must be a sequence of 3 floats")

        if isinstance(max_degrees, (float, int)):
            self.max_degrees = torch.tensor([max_degrees, max_degrees, max_degrees])
        elif isinstance(max_degrees, Sequence):
            self.max_degrees = torch.tensor(max_degrees)
            if self.max_degrees.shape != (3,):
                raise ValueError(
                    "max_degrees must be a float or a sequence of 3 floats"
                )
        elif max_degrees is None:
            self.max_degrees = None

        if (
            self.max_degrees is not None
            and not ((0 <= self.max_degrees) & (self.max_degrees <= 180)).all().item()
        ):
            raise ValueError("max_degrees must be in [0, 180]")

        # Convert to radians
        if self.max_degrees is not None:
            self.max_degrees = self.max_degrees * torch.pi / 180.0

    def random_pose(self, device: str | torch.device) -> Tensor:
        if self.max_degrees is None:
            quat = random_orientation(num=1, device=device)
            rot = quaternion_to_matrix(quat)
        else:
            random_euler = sample_uniform(
                lower=-self.max_degrees,
                upper=self.max_degrees,
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

        transform = make_pose(translation, rot).squeeze(0)
        return transform


class RandomlyTransformPointMapTrajectory(BaseRandomlyTransform):
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
        super().__init__(max_translation, max_angle)

        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        device = tensordict.device or "cpu"

        transform = self.random_pose(device)

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
