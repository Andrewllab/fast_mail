from __future__ import annotations

import torch
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import Transform
from utils.math import combine_frame_transforms, subtract_frame_transforms


class SubsampleTime(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        factor: int,
    ):
        self._specs = specs
        self._factor = factor

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # convert delta ee pose actions to a trajectory of desired end effector
        # poses
        delta_ee_pose = tensordict["action"][..., :-1]
        current_ee_pose = tensordict["obs", "ee_pose"]
        desired_ee_pos, desired_ee_rot = combine_frame_transforms(
            current_ee_pose[..., :3],
            current_ee_pose[..., 3:7],
            delta_ee_pose[..., :3],
            delta_ee_pose[..., 3:7],
        )
        tensordict["target_ee_pose"] = torch.cat(
            (desired_ee_pos, desired_ee_rot), dim=-1
        )

        # compute an offset, since we want to keep the very last time step, and
        # rather remove some more initial frames
        offset = (tensordict.shape[0] - 1) % self._factor
        # subsample the trajectory
        tensordict = tensordict[offset :: self._factor]

        # convert desired ee pose trajectory back to delta ee pose actions
        desired_ee_pose = tensordict["target_ee_pose"]
        current_ee_pose = tensordict["obs", "ee_pose"]
        delta_ee_pos, delta_ee_rot = subtract_frame_transforms(
            current_ee_pose[..., :3],
            current_ee_pose[..., 3:7],
            desired_ee_pose[..., :3],
            desired_ee_pose[..., 3:7],
        )

        # TODO: the gripper actions are simply subsampled with no smoothing
        # or interpolation. Is this a problem?
        tensordict["action"] = torch.cat(
            (
                delta_ee_pos,
                delta_ee_rot,
                tensordict["action"][..., -1:],
            ),
            dim=-1,
        )

        return tensordict
