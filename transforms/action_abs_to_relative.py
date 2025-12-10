from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import ReversibleTransform, TransformConstraint
from utils.math import combine_frame_transforms, normalize, subtract_frame_transforms


class AbsoluteTaskActionsToRelativeChunk(ReversibleTransform):
    """Preprocess transforms for converting actions (target ee_poses) from
    the world (robot base) frame into action chunks, where the actions in each
    chunk are relative to their own reference frame. This reference frame can
    be the current ee_pose (where current means at the first time step of the
    chunk) or the last action (target ee_pose) (where last means the time step
    prior to the beginning of the chunk).
    """

    constraints = [TransformConstraint.TRAJECTORY_ONLY]

    def __init__(
        self,
        specs: DataSpecs,
        reference: Literal["current_ee_pose", "last_action"],
        # we could get this from the specs, but by putting action_seq_len in
        # the arguments, we trigger new preprocessing if it changes
        # BackCompat: allow None for getting action_seq_len from specs
        action_seq_len: int | None = None,
    ):
        self._specs = specs
        self.action_seq_len = (
            action_seq_len if action_seq_len is not None else specs.action_seq_len
        )
        self.reference = reference
        if reference != "current_ee_pose":
            raise NotImplementedError(
                "Only 'current_ee_pose' reference is implemented for now."
            )

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reference={self.reference})"

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:

        absolute_action = tensordict["action"]
        # since each chunk will be normalized to a difference reference, we
        # need to explicitly construct them
        absolute_chunks = absolute_action.unfold(
            0,  # dimension to unfold
            self.action_seq_len,  # size of each chunk
            1,  # step
        )
        # move the chunk dimension to the second position
        absolute_chunks = absolute_chunks.movedim(-1, 1)

        # since subtract_frame_transforms can only handle one leading dimension, flatten
        absolute_chunks_flat = absolute_chunks.flatten(start_dim=0, end_dim=1)
        abs_pos, abs_quat = (
            absolute_chunks_flat[..., :3],
            absolute_chunks_flat[..., 3:7],
        )

        # current_ee_pose: [T, 7]
        current_ee_pose = tensordict["obs", "ee_pose"]
        # trim the current_ee_pose at the end of the trajectory, which do not occur at the beginning of a chunk
        n_chunks = absolute_chunks.shape[0]
        current_ee_pose = current_ee_pose[:n_chunks]
        # repeat the current_ee_pose for each action in the chunk
        current_ee_pose_flat = current_ee_pose.repeat_interleave(
            repeats=self.action_seq_len, dim=0
        )
        ref_pos, ref_quat = (
            current_ee_pose_flat[..., :3],
            current_ee_pose_flat[..., 3:7],
        )

        # subtract the reference pose from the absolute pose to get the relative action
        # order of arguments is reversed from what is expected
        rel_pos, rel_quat = subtract_frame_transforms(
            ref_pos, ref_quat, abs_pos, abs_quat
        )

        # concatenate the relative position, relative quaternion, and gripper command
        # and reshape
        gripper_command = absolute_chunks_flat[..., 7:]
        rel_action_flat = torch.cat((rel_pos, rel_quat, gripper_command), dim=-1)
        rel_action = rel_action_flat.unflatten(
            dim=0, sizes=(n_chunks, self.action_seq_len)
        )

        tensordict["action"] = rel_action

        # TODO: slice obs to match the new action shape

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # This gets called when running with an environment, where the
        # preprocess transforms get rolled into the other transforms.
        # We don't need to do anything, since there are no actions to transform.
        assert not self.training
        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:

        rel_action = tensordict["action"]
        B, action_seq_len = rel_action.shape[:2]

        # since combine_frame_transforms can only handle one leading dimension, flatten
        rel_action_flat = rel_action.flatten(start_dim=0, end_dim=1)
        rel_pos, rel_quat = (
            rel_action_flat[..., :3],
            rel_action_flat[..., 3:7],
        )
        rel_quat = normalize(rel_quat)

        # current_ee_pose: [B, 1, 7]
        current_ee_pose = tensordict["obs", "ee_pose"]
        # repeat the current_ee_pose for each action in the chunk
        current_ee_pose = current_ee_pose.repeat_interleave(
            repeats=action_seq_len, dim=1
        )
        current_ee_pose_flat = current_ee_pose.flatten(start_dim=0, end_dim=1)
        ref_pos, ref_quat = (
            current_ee_pose_flat[..., :3],
            current_ee_pose_flat[..., 3:7],
        )

        # add the reference pose to the relative pose to get the absolute action
        abs_pos, abs_quat = combine_frame_transforms(
            ref_pos, ref_quat, rel_pos, rel_quat
        )

        # concatenate the relative position, relative quaternion, and gripper command
        # and reshape
        gripper_command = rel_action_flat[..., 7:]
        abs_action_flat = torch.cat((abs_pos, abs_quat, gripper_command), dim=-1)
        abs_action = abs_action_flat.unflatten(dim=0, sizes=(B, action_seq_len))

        tensordict["action"] = abs_action

        return tensordict


class AbsoluteTaskActionsFrameTransform(ReversibleTransform, nn.Module):
    """Preprocess transforms for converting actions (target ee_poses) from
    the world (robot base) frame to be relative to some reference frame. This
    reference frame can be the first ee_pose in the trajectory or the first
    action (target ee_poses) in the trajectory.
    """

    constraints = [TransformConstraint.TRAJECTORY_ONLY]
    ref_pose: torch.Tensor

    def __init__(
        self,
        specs: DataSpecs,
        reference: Literal["first_ee_pose", "first_action"],
    ):
        super().__init__()

        self._specs = specs
        self.reference = reference
        self.first = True  # flag to indicate if we are processing the first step
        self.register_buffer("ref_pose", torch.zeros((1, 7), dtype=torch.float32))

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reference={self.reference})"

    def forward(self, tensordict: TensorDict) -> TensorDict:
        # This gets called when running with an environment, where the
        # preprocess transforms get rolled into the other transforms.
        # We don't need to do anything, since there are no actions to transform.
        assert not self.training
        return tensordict

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:

        absolute_action = tensordict["action"]
        assert absolute_action.ndim == 2  # [T, 8]
        assert absolute_action.shape[1] == 8
        T = absolute_action.shape[0]

        # ref: [1, 7]
        if self.reference == "first_ee_pose":
            ref = tensordict["obs", "ee_pose"][:1]
        elif self.reference == "first_action":
            ref = absolute_action[:1, :7]
        else:
            raise ValueError(f"Unknown reference: {self.reference}")

        # repeat the reference for each action in the chunk
        ref = ref.repeat_interleave(repeats=T, dim=0)  # ref: [T, 7]

        abs_pos, abs_quat = (
            absolute_action[..., :3],
            absolute_action[..., 3:7],
        )
        ref_pos, ref_quat = (
            ref[..., :3],
            ref[..., 3:7],
        )

        rel_pos, rel_quat = subtract_frame_transforms(
            ref_pos, ref_quat, abs_pos, abs_quat
        )

        gripper_command = absolute_action[..., 7:]
        rel_action = torch.cat((rel_pos, rel_quat, gripper_command), dim=-1)
        tensordict["action"] = rel_action

        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:

        if self.first:
            # store the first ee_pose or first action as the reference

            if self.reference == "first_ee_pose":
                ref = tensordict["obs", "ee_pose"]
            elif self.reference == "first_action":
                ref = tensordict["action"][..., :7]
            else:
                raise ValueError(f"Unknown reference: {self.reference}")

            assert ref.ndim == 3  # [B, T, 7]
            # batch size must be 1, otherwise we would need to know how many refs to allocate
            assert ref.shape[0] == 1
            assert ref.shape[2] == 7  # 7 elements: [x, y, z, qw, qx, qy, qz]
            self.ref_pose[...] = ref[0, 0]
            self.first = False

        rel_action = tensordict["action"]
        assert rel_action.ndim == 3  # [B, T, 8]
        assert rel_action.shape[0] == 1
        assert rel_action.shape[2] == 8
        action_seq_len = rel_action.shape[0]
        leading_dims = rel_action.shape[:-1]
        rel_action = rel_action.flatten(start_dim=0, end_dim=-2)

        # self.ref_pose: [1, 7]

        # repeat the reference for each action in the chunk
        ref = self.ref_pose.repeat_interleave(repeats=action_seq_len, dim=0)

        rel_pos, rel_quat = (
            rel_action[..., :3],
            rel_action[..., 3:7],
        )
        ref_pos, ref_quat = (
            ref[..., :3],
            ref[..., 3:7],
        )

        abs_pos, abs_quat = combine_frame_transforms(
            ref_pos, ref_quat, rel_pos, rel_quat
        )

        gripper_command = rel_action[..., 7:]
        abs_action = torch.cat((abs_pos, abs_quat, gripper_command), dim=-1)
        abs_action = abs_action.unflatten(dim=0, sizes=leading_dims)
        tensordict["action"] = abs_action

        return tensordict


class AbsoluteJointActionsToDelta(ReversibleTransform):
    """Preprocess transform for converting actions (target joint positions)
    from absolute positions to deltas relative to the current joint positions.
    """

    constraints = [TransformConstraint.TRAJECTORY_ONLY]

    def __init__(
        self,
        specs: DataSpecs,
        reference: Literal["joint_pos", "target_joint_pos"],
        joint_pos_key: str = "joint_pos",
        target_joint_pos_key: str = "target_joint_pos",
    ):
        if reference not in ["joint_pos", "target_joint_pos"]:
            raise ValueError(f"Unknown reference: {reference}")

        if target_joint_pos_key not in specs.obs:
            raise ValueError(
                f"Target joint position key '{target_joint_pos_key}' not found in specs."
            )

        self._specs = specs
        self.reference = reference
        self.joint_pos_key = joint_pos_key
        self.target_joint_pos_key = target_joint_pos_key

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:

        action = tensordict["action"]
        target_joint_pos = action[..., :-1]  # exclude gripper command

        # verify that target_joint_pos in the action matches the one in the obs
        target_joint_pos2 = tensordict["obs", self.target_joint_pos_key]
        assert torch.allclose(target_joint_pos, target_joint_pos2)
        current_joint_pos = tensordict["obs", self.joint_pos_key]

        if self.reference == "joint_pos":
            delta_joint_pos = target_joint_pos - current_joint_pos

        elif self.reference == "target_joint_pos":
            delta_joint_pos = torch.zeros_like(target_joint_pos)
            delta_joint_pos[1:] = target_joint_pos[1:] - target_joint_pos[:-1]

            # # for the first time step, use the current joint positions as the
            # # reference
            # delta_joint_pos[0] = target_joint_pos[0] - current_joint_pos[0]

            # we leave the first delta as zero, since there is a sizeable offset
            # between target and current joint positions at the first time step

        else:
            raise ValueError(f"Unknown reference: {self.reference}")

        action[..., :-1] = delta_joint_pos

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # This gets called when running with an environment, where the
        # preprocess transforms get rolled into the other transforms.
        # We don't need to do anything, since there are no actions to transform.
        assert not self.training
        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:

        action = tensordict["action"]
        assert action.ndim == 3  # [B, T, num_joints]
        delta_joint_pos = action[..., :-1]  # exclude gripper command

        # accumulate the deltas across the action chunk
        delta_chunk = delta_joint_pos.cumsum(dim=1)

        if self.reference == "joint_pos":
            current_joint_pos = tensordict["obs", self.joint_pos_key]
            assert current_joint_pos.ndim == 3  # [B, T, num_joints]

            # broadcast current_joint_pos to match delta_chunk shape
            target_joint_pos = delta_chunk + current_joint_pos[:, -1:]

        elif self.reference == "target_joint_pos":
            current_target_pos = tensordict["obs", self.target_joint_pos_key]
            assert current_target_pos.ndim == 3  # [B, T, num_joints]

            # broadcast current_joint_pos to match delta_chunk shape
            target_joint_pos = delta_chunk + current_target_pos

        else:
            raise ValueError(f"Unknown reference: {self.reference}")

        action[..., :-1] = target_joint_pos

        return tensordict
