from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import ReversibleTransform, TransformConstraint
from utils.math import combine_frame_transforms, normalize, subtract_frame_transforms

log = logging.getLogger(__name__)


class AbsoluteEeActionsToRelativeChunk(ReversibleTransform):
    """Preprocess transforms for converting actions (target ee_poses) from
    the world (robot base) frame into action chunks, where the actions in each
    chunk are relative to their own reference frame. This reference frame can
    be the current ee_pose (where current means at the first time step of the
    chunk) or the last action (target ee_pose) (where last means the time step
    prior to the beginning of the chunk).
    """

    constraints = [
        # Cannot be applied to data from env, as the forward call modifies actions
        TransformConstraint.DATASET_ONLY,
        # Must be applied on trajectories because we need to chunk the actions
        TransformConstraint.TRAJECTORY_ONLY,
    ]

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


class AbsoluteEeActionsToDelta(ReversibleTransform):
    # Cannot be applied to data from env, as the forward call modifies actions
    constraints = [TransformConstraint.DATASET_ONLY]

    def __init__(self, specs: DataSpecs, ref_pose_obs_key: str):
        if ref_pose_obs_key not in specs.obs:
            raise ValueError(f"Key '{ref_pose_obs_key}' not found in obs specs.")

        action_spec = specs.action
        # This transform assumes 8D actions: 3 for position, 4 for quaternion,
        # 1 for gripper
        # TODO: more graceful handling of different action specs
        assert action_spec.action_dim == 8

        ref_pose_spec = specs.obs[ref_pose_obs_key]
        if ref_pose_spec.shape[-1] != 7:
            raise ValueError(
                f"Expected element shape of reference ee pose at 'obs/{ref_pose_obs_key}' to be 7, but got {ref_pose_spec.shape[-1]}."
            )

        self._specs = specs
        self.ref_pose_obs_key = ref_pose_obs_key

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reference_pose=obs/{self.ref_pose_obs_key})"

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"]
        ref_ee_pose = tensordict["obs", self.ref_pose_obs_key]

        # TODO: changed to warning because TrimIdle removes some time steps.
        # This causes misalignment because the `ref_ee_pose` references
        # whatever used to be the last step, which may have changed.
        # Once TrimIdle updates fields that are previous time steps, raise an
        # error here again.
        if not torch.allclose(action[:-1, ..., :-1], ref_ee_pose[1:]):
            log.warning(
                f"The values at `obs/{self.ref_pose_obs_key}` need to contain the action (i.e. the target ee pose) from the previous step. "
                "This is required so that the transform behaves the same way during rollout on the environment."
            )

        # delta_t = action_t - action_{t-1}
        #         = action_t - target_ee_pose_t
        # This is important so that we don't need values from future or
        # past time steps to reconstruct the target ee pose, which we don't
        # have during rollout.
        delta_pos, delta_quat = subtract_frame_transforms(
            ref_ee_pose[..., :3],
            ref_ee_pose[..., 3:7],
            action[..., :3],
            action[..., 3:7],
        )

        # write new actions in place
        action[..., :3] = delta_pos
        action[..., 3:7] = delta_quat

        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"]
        assert action.ndim == 3  # [B, T, 8]
        assert action.shape[-1] == 8
        B, T = action.shape[:2]
        delta_pos, delta_quat = action[..., :3], action[..., 3:7]

        # get the starting ee_pose for each batch
        ref_ee_pose = tensordict["obs", self.ref_pose_obs_key]
        assert ref_ee_pose.ndim == 3  # [B, T, 7]
        assert ref_ee_pose.shape[-1] == 7
        ref_ee_pose = ref_ee_pose[:, -1]  # get the last ee_pose for each batch

        # accumulate the deltas across the action chunk
        abs_ee_pos = torch.zeros_like(delta_pos)
        abs_ee_quat = torch.zeros_like(delta_quat)
        abs_ee_pos[:, 0], abs_ee_quat[:, 0] = combine_frame_transforms(
            ref_ee_pose[:, :3], ref_ee_pose[:, 3:7], delta_pos[:, 0], delta_quat[:, 0]
        )
        for t in range(1, T):
            abs_ee_pos[:, t], abs_ee_quat[:, t] = combine_frame_transforms(
                abs_ee_pos[:, t - 1],
                abs_ee_quat[:, t - 1],
                delta_pos[:, t],
                delta_quat[:, t],
            )

        # normalize again, in case repeated quaternion multiplications caused drift
        # TODO: replace with check and log any quats that are not normalized
        abs_ee_quat = normalize(abs_ee_quat)

        # write reversed actions in place
        action[..., :3] = abs_ee_pos
        action[..., 3:7] = abs_ee_quat

        return tensordict


class AbsoluteEeActionsFrameTransform(ReversibleTransform, nn.Module):
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
        raise NotImplementedError(
            "This transform is currently broken, due to how it handles state."
        )

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

    # Cannot be applied to data from env, as the forward call modifies actions
    constraints = [TransformConstraint.DATASET_ONLY]

    def __init__(self, specs: DataSpecs, ref_pos_obs_key: str):

        if ref_pos_obs_key not in specs.obs:
            raise ValueError(f"Key '{ref_pos_obs_key}' not found in obs specs.")

        action_spec = specs.action
        n_joints = action_spec.action_dim - 1  # exclude gripper command

        ref_pose_spec = specs.obs[ref_pos_obs_key]
        if ref_pose_spec.shape[-1] != n_joints:
            raise ValueError(
                f"Expected element shape of reference joint positions at 'obs/{ref_pos_obs_key}' to be {n_joints}, but got {ref_pose_spec.shape[-1]}."
            )

        self._specs = specs
        self.n_joints = n_joints
        self.ref_pos_obs_key = ref_pos_obs_key

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reference_pos=obs/{self.ref_pos_obs_key})"

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"]
        ref_joint_pos = tensordict["obs", self.ref_pos_obs_key]

        # TODO: changed to warning because TrimIdle removes some time steps.
        # This causes misalignment because the `ref_joint_pos` references
        # whatever used to be the last step, which may have changed.
        # Once TrimIdle updates fields that are previous time steps, raise an
        # error here again.
        if not torch.allclose(action[:-1, ..., :-1], ref_joint_pos[1:]):
            log.warning(
                f"The values at `obs/{self.ref_pos_obs_key}` need to contain the action (i.e. the target joint positions) from the previous step. "
                "This is required so that the transform behaves the same way during rollout on the environment."
            )

        # delta_t = action_t - action_{t-1}
        #         = action_t - target_joint_pos_t
        # This is important so that we don't need values from future or
        # past time steps to reconstruct the target joint positions,
        # which we don't have during rollout.
        delta_joint_pos = action[..., :-1] - ref_joint_pos

        # write new actions in place
        action[..., :-1] = delta_joint_pos

        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"]
        assert action.ndim == 3  # [B, T, num_joints + 1]
        assert action.shape[-1] == self.n_joints + 1
        delta_joint_pos = action[..., :-1]  # exclude gripper command

        # get the starting joint positions for each batch
        ref_joint_pos = tensordict["obs", self.ref_pos_obs_key]
        assert ref_joint_pos.ndim == 3  # [B, T, num_joints]
        assert ref_joint_pos.shape[-1] == self.n_joints
        ref_joint_pos = ref_joint_pos[:, -1:]  # get the last joint pos for each batch

        # accumulate the deltas across the time dim of the action chunk
        delta_chunk = delta_joint_pos.cumsum(dim=1)

        # add the deltas to the current reference joint pos (broadcasts
        # across time dimension)
        target_joint_pos = ref_joint_pos + delta_chunk

        # write reversed actions in place
        action[..., :-1] = target_joint_pos

        return tensordict
