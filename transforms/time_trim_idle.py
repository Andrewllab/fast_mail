from __future__ import annotations

import logging

import torch
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import Transform, TransformConstraint
from utils.math import axis_angle_from_quat, subtract_frame_transforms

log = logging.getLogger(__name__)


class TrimIdleStart(Transform):
    constraints = [TransformConstraint.TRAJECTORY_ONLY]

    def __init__(
        self,
        specs: DataSpecs,
        pos_threshold: float = 1e-6,
        rot_threshold: float = 1e-5,
        margin: int = 0,
    ):
        self._specs = specs
        self.pos_threshold = pos_threshold
        self.rot_threshold = rot_threshold
        self.margin = margin

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        absolute_action = tensordict["action"]
        absolute_pos = absolute_action[..., :3]
        absolute_quat = absolute_action[..., 3:7]

        delta_pos, delta_quat = subtract_frame_transforms(
            absolute_pos[:-1], absolute_quat[:-1], absolute_pos[1:], absolute_quat[1:]
        )

        delta_axis_angle = axis_angle_from_quat(delta_quat)

        is_moving = torch.logical_or(
            delta_pos.norm(dim=-1) > self.pos_threshold,
            delta_axis_angle.norm(dim=-1) > self.rot_threshold,
        )

        start = is_moving.to(torch.int32).argmax(dim=-1)
        start = (start - self.margin).clamp(min=0)
        # TODO: also trim the end of the trajectory

        log.debug(
            f"Trimming the first {start} steps from trajectory of length {len(is_moving)}."
        )

        # TODO: slice anything with a leading dimension of T
        tensordict["obs"] = tensordict["obs"][start:]
        tensordict["action"] = tensordict["action"][start:]
        tensordict["ref_action"] = tensordict["ref_action"][start:]

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # This gets called when running with an environment, where the
        # preprocess transforms get rolled into the cpu_batch_transforms.
        # We don't need to do anything, since we can only run during
        # preprocessing.
        return tensordict
