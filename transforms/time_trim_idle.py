from __future__ import annotations

import logging
from typing import Literal, Sequence

import torch
from tensordict import TensorDict

from environments.datamodule import EmptyTrajectoryError
from environments.specs import DataSpecs
from transforms.base_transform import Transform, TransformConstraint
from utils.math import axis_angle_from_quat, subtract_frame_transforms

log = logging.getLogger(__name__)

COMPARE_TYPE = Literal["joint_pos", "action"]
TRIM_TYPE = Literal["start", "inside", "end"]


class TrimIdle(Transform):
    """
        Trim idle time steps from trajectories based on action movement.

    Args:
        specs (DataSpecs): The data specifications.
        action_type (Literal["joint", "cartesian"]): The type of action space.
        pos_threshold (float): The position change threshold to consider as movement.
        rot_threshold (float): The rotation change threshold to consider as movement.
        gripper_threshold (float): The gripper position change threshold to consider as movement.
        trim_type (Literal["start", "all", "inside", "end"]): The type of trimming to perform. "start" trims only
            the start of the trajectory, "all" trims any idle span that is at least window_size long, inside trims
            idle spans but keeps the start and end, "end" trims only the end of the trajectory.
        compare (Sequence[Literal["joint_pos", "action"]]): Which data to use for movement comparison.
        window_size (int): The size of the sliding window to check for movement.
        margin (int): Keep this many idle frames at the beginning when using "start" trim_type.
        error_empty_trajectories (bool): If True, raise an error if the resulting trajectory is empty.
    """

    constraints = [TransformConstraint.TRAJECTORY_ONLY]

    def __init__(
        self,
        specs: DataSpecs,
        compare: COMPARE_TYPE | Sequence[COMPARE_TYPE],
        trim_type: TRIM_TYPE | Sequence[TRIM_TYPE],
        pos_threshold: float,
        gripper_threshold: float,
        rot_threshold: float | None = None,
        action_type: Literal["joint", "cartesian"] | None = None,
        window_size: int = 1,
        margin: int = 0,
        error_empty_trajectories: bool = True,
    ):
        if isinstance(compare, str):
            compare = [compare]

        self.compare_action = "action" in compare
        self.compare_state = "joint_pos" in compare

        if not self.compare_action and not self.compare_state:
            raise ValueError(
                "At least one of 'action' or 'joint_pos' must be in compare."
            )

        if isinstance(trim_type, str):
            trim_type = [trim_type]

        for t in trim_type:
            if t not in ["start", "inside", "end"]:
                raise ValueError(f"Invalid trim_type: {t}")

        if self.compare_action and action_type not in ["joint", "cartesian"]:
            raise ValueError(f"Invalid action_type: {action_type}")

        if action_type == "cartesian" and rot_threshold is None:
            raise ValueError("rot_threshold must be provided for cartesian action_type")

        self._specs = specs
        self.trim_type = trim_type
        self.pos_threshold = pos_threshold
        self.gripper_threshold = gripper_threshold
        self.rot_threshold = rot_threshold
        self.action_type = action_type
        self.window_size = window_size
        self.margin = margin
        self.error_empty_trajectories = error_empty_trajectories

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:

        T = tensordict["obs"].size(0)
        is_moving = torch.zeros(T - 1, dtype=torch.bool, device=tensordict.device)

        if self.compare_state:
            robot_state = tensordict["obs"]["robot_state"]
            joint_pos = robot_state[..., :-1]  # Exclude gripper state

            # check joint position changes
            delta_state = joint_pos[1:] - joint_pos[:-1]
            is_moving = torch.logical_or(
                is_moving, delta_state.norm(dim=-1) > self.pos_threshold
            )

            # check gripper position changes
            gripper_pos = robot_state[..., -1]
            delta_gripper_pos = gripper_pos[1:] - gripper_pos[:-1]
            is_moving = torch.logical_or(
                is_moving,
                delta_gripper_pos.abs() > self.gripper_threshold,
            )

        if self.compare_action:
            action = tensordict["action"]

            if self.action_type == "cartesian":
                target_ee_pos = action[..., :3]
                target_ee_quat = action[..., 3:7]

                delta_pos, delta_quat = subtract_frame_transforms(
                    target_ee_pos[:-1],
                    target_ee_quat[:-1],
                    target_ee_pos[1:],
                    target_ee_quat[1:],
                )

                is_moving = torch.logical_or(
                    is_moving, delta_pos.norm(dim=-1) > self.pos_threshold
                )

                delta_axis_angle = axis_angle_from_quat(delta_quat)
                is_moving = torch.logical_or(
                    is_moving, delta_axis_angle.norm(dim=-1) > self.rot_threshold
                )

            elif self.action_type == "joint":
                target_joint_pos = action[..., :-1]
                delta_action = target_joint_pos[1:] - target_joint_pos[:-1]
                is_moving = torch.logical_or(
                    is_moving, delta_action.norm(dim=-1) > self.pos_threshold
                )

            # Check gripper action
            target_gripper_pos = action[..., -1]
            delta_gripper_pos = target_gripper_pos[1:] - target_gripper_pos[:-1]
            is_moving = torch.logical_or(
                is_moving,
                delta_gripper_pos.abs() > self.gripper_threshold,
            )

        # Handle case where no valid segments are found
        if is_moving.sum() == 0:
            log.warning(
                "No movement detected in trajectory %sfrom %s. Returning trajectory unchanged.",
                (f"named {tensordict['name']} " if "name" in tensordict else ""),
                tensordict["path"],
            )
            if self.error_empty_trajectories:
                raise EmptyTrajectoryError("Aborting due to empty trajectory.")
            return tensordict

        start, end = is_moving.nonzero(as_tuple=True)[0][[0, -1]]
        end += 1  # include last moving step

        if "inside" in self.trim_type:
            # Check if there is enough movement in all sliding windows of size `window_size`
            is_moving_window = is_moving.unfold(0, self.window_size, 1).any(dim=-1)

            # Construct a [T, window_size] tensor of all indices contained in each window
            idx = torch.arange(
                is_moving_window.size(0), device=is_moving.device
            ).unsqueeze(1) + torch.arange(self.window_size, device=is_moving.device)

            # Find all indices that belong to a window without any movement.
            # Indices may be present multiple times, which is fine.
            # Since is_moving is T-1 long, we effectively keep the last frame
            # of each non-moving span.
            drop_idx = idx[~is_moving_window].flatten()

            # Create a boolean mask of frames to keep.
            mask = torch.ones(T, dtype=torch.bool, device=action.device)
            # Set indices to drop to False (duplicate indices are fine).
            mask[drop_idx] = False

            # do not trim the starting and ending frames, we'll handle those separately
            mask[:start] = True
            mask[end:] = True

            log.debug(
                f"Trimming {mask.numel() - mask.sum()} idle steps from trajectory of length {len(is_moving) + 1}."
            )

            # TODO: slice anything with a leading dimension of T
            tensordict["obs"] = tensordict["obs"][mask]
            tensordict["action"] = tensordict["action"][mask]
            tensordict["ref_action"] = tensordict["ref_action"][mask]

            is_moving = is_moving[mask[:-1]]  # update is_moving for start/end trimming

        # update T, start and end after inside trimming
        T = tensordict["obs"].size(0)
        start, end = is_moving.nonzero(as_tuple=True)[0][[0, -1]]
        end += 1  # include last moving step

        # trim end before start so that the indices remain valid
        end = (end + self.margin).clamp(max=T)
        if "end" in self.trim_type and end < T - 1:
            log.debug(
                f"Trimming the last {(T - end).item()} steps from trajectory of length {len(is_moving) + 1}."
            )

            # TODO: slice anything with a leading dimension of T
            tensordict["obs"] = tensordict["obs"][:end]
            tensordict["action"] = tensordict["action"][:end]
            tensordict["ref_action"] = tensordict["ref_action"][:end]

        start = (start - self.margin).clamp(min=0)
        if "start" in self.trim_type and start > 0:
            log.debug(
                f"Trimming the first {start.item()} steps from trajectory of length {len(is_moving) + 1}."
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
