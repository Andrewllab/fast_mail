from __future__ import annotations

import dataclasses
import logging
from typing import Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import ReversibleTransform, TransformConstraint
from utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    convert_quat,
    matrix_to_quaternion,
    normalize,
    quat_from_axis_angle,
    quaternion_to_matrix,
    subtract_frame_transforms,
    transform_points,
)
from utils.nested import to_strided_tensor

log = logging.getLogger(__name__)


POINTS_LOCAL = torch.tensor(
    [
        [0.05, 0.0, -0.05],
        [-0.05, 0.0, -0.05],
        [0.0, 0.00, -0.1],
    ]  # ( x, y, z)
)

POINTS_FINGERS = torch.tensor(
    [
        [[0.005, 0.0, 0.00], [-0.005, 0.0, 0.00]],
        [[0.005, 0.0, 0.00], [-0.005, 0.0, 0.00]],
        [[0.05, 0.0, 0.00], [-0.05, 0.0, 0.00]],
    ]
)


class AbsoluteEEPoseToFivePointsTransform(ReversibleTransform):
    """Preprocess transforms for converting actions (target ee_poses) from
    the world (robot base) frame to be relative to some reference frame. This
    reference frame can be the first ee_pose in the trajectory or the first
    action (target ee_poses) in the trajectory.
    """

    def __init__(
        self,
        specs: DataSpecs,
        ee_pose_key: str = "ee_pose",
    ):
        super().__init__()

        self._specs = specs
        self.ee_pose_key = ee_pose_key

        action_spec = specs.action
        # TODO: more graceful handling of different action specs
        new_action_spec = dataclasses.replace(action_spec, action_dim=5 * 3)

        self._specs = specs.replace(action=new_action_spec)

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        # Action from pos rot to 5 3D points
        action_pos = tensordict["action"][..., :3]  # (T, 3)
        action_rot = tensordict["action"][..., 3:6]  # (T, 3)
        gripper_action = tensordict["action"][..., 6:]  # (T, 1)

        action_points = self.ee_pose_to_3D_points(
            ee_pos=action_pos, ee_rot=action_rot, gripper=gripper_action
        )  # (T, 5, 3)

        # Transform action points to world frame
        base_pose = tensordict["obs"]["base_pose"]  # (T, 7)
        base_pos = base_pose[..., :3]  # (T, 3)
        base_quat = base_pose[..., 3:]  # (T, 4)
        action_points = transform_points(action_points, base_pos, base_quat)

        # Compute current points
        current_pos = tensordict["obs"][self.ee_pose_key][..., :3]  # (T, 3)
        current_quat = tensordict["obs"][self.ee_pose_key][..., 3:]  # (T, 4)
        current_gripper = -torch.ones_like(gripper_action)
        current_gripper[1:] = tensordict["action"][:-1, 6:]  # (T, 1)
        current_points = self.ee_pose_to_3D_points(
            ee_pos=current_pos,
            ee_quat=current_quat,
            gripper=current_gripper,
        )  # (T, 5, 3)

        tensordict["action"] = action_points
        tensordict["obs"]["gripper_points"] = {"points": current_points}

        return tensordict

    def call_trajectory_rollout(self, tensordict: TensorDict) -> TensorDict:
        ee_pos = tensordict["obs"]["ee_pose"][:, :3].to(torch.float32)  # (T, 3)
        ee_quat = tensordict["obs"]["ee_pose"][:, 3:].to(torch.float32)  # (T, 4)
        if "_action" not in tensordict["obs"].keys():
            gripper_state = -torch.ones(
                ee_pos.shape[0], device=ee_pos.device, dtype=ee_pos.dtype
            )  # (T, 1)
        else:
            gripper_state = tensordict["obs"]["_action"][:, 6]  # (T,)

        points = self.ee_pose_to_3D_points(
            ee_pos=ee_pos, ee_quat=ee_quat, gripper=gripper_state
        )  # (T, 5, 3)

        base_pose = tensordict["obs"]["base_pose"]  # (T, 7)
        base_pos = base_pose[..., :3]  # (T, 3)
        base_quat = base_pose[..., 3:]  # (T, 4)
        points = transform_points(points, base_pos, base_quat)

        tensordict["obs"]["gripper_points"] = {"points": points[0].to(torch.float32)}

        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        points = tensordict["action"].float()  # (B, 5, T, 3)
        if points.is_nested:
            if (torch.diff(points.offsets()) == 5).all():
                points = to_strided_tensor(points).unsqueeze(2)
            else:
                raise ValueError(
                    "Cannot reverse transform: action points tensor is jagged with unexpected offsets."
                )
        B, N, T, _ = points.shape
        points = points.swapaxes(1, 2).reshape(-1, 5, 3)  # (B*T, 5, 3)

        base_pose = tensordict["obs"]["base_pose"]  # (T, 7)
        base_pos = base_pose[..., 0, :3]  # (T, 3)
        base_quat = base_pose[..., 0, 3:]  # (T, 4)
        inv_base_pos, inv_base_quat = subtract_frame_transforms(base_pos, base_quat)
        points = transform_points(points, inv_base_pos, inv_base_quat)

        ee_pos, ee_rot = self.points_to_pose(points[:, :3])  # (T, 3), (T, 3)
        gripper = self.points_to_gripper(points[:, 3:5])  # (T,)
        ee_poses = torch.cat([ee_pos, ee_rot], dim=-1)  # (T, 6)
        # Convert back to original batched shape
        ee_poses = ee_poses.view(B, T, 6)
        gripper = gripper.view(B, T)

        tensordict["action"] = torch.cat(
            [ee_poses, gripper.unsqueeze(-1)], dim=-1
        )  # (T, 8)

        return tensordict

    @torch.no_grad()
    def points_to_pose(self, points_global):
        """
        Recovers the translation (ee_pos) and rotation (ee_quat) for each batch sample.

        Arguments:
        points_local: [N, 3] tensor of fixed local points.
        points_global: [B, N, 3] tensor of corresponding global points for each batch.

        Returns:
        ee_pos: [B, 3] tensor of recovered translations.
        ee_quat: [B, 4] tensor of recovered quaternions (in [qx, qy, qz, qw] format).
        """
        points_local = POINTS_LOCAL.to(points_global.device)

        T, N, _ = points_global.shape

        # Compute the centroid of the local points (constant for all batches)
        centroid_local = points_local.mean(dim=0, keepdim=True)  # [1, 3]
        X = points_local - centroid_local  # [N, 3] centered local points

        # Compute centroids for each batch sample
        centroid_global = points_global.mean(dim=-2, keepdim=True)  # [B, 1, 3]
        Y = points_global - centroid_global  # [B, N, 3] centered global points

        # Compute covariance matrices for each batch sample: H = X^T @ Y
        # Expand X^T to batch: [B, 3, N]
        X_T = X.t().unsqueeze(0).expand(T, 3, N)
        H = torch.bmm(X_T, Y)  # , out_dtype=X_T.dtype)  # [B, 3, 3]

        # Compute SVD for each batch sample
        U, S, Vh = torch.linalg.svd(H.to(torch.float32), full_matrices=False)
        V = Vh.transpose(-2, -1)  # [B, 3, 3]

        # Compute rotation: R = V @ U^T for each batch
        R = torch.bmm(V, U.transpose(-2, -1))  # , out_dtype=V.dtype)  # [B, 3, 3]

        # Reflection correction: if det(R) < 0, flip sign of last column of V for that sample.
        det_R = torch.det(R.to(torch.float32))  # [B]
        for i in range(T):
            if det_R[i] < 0:
                V[i, :, -1] *= -1
        R = torch.bmm(V, U.transpose(-2, -1))  # [B, 3, 3]

        # Compute translation: T = centroid_global - R @ centroid_local^T, for each batch sample.
        # Expand centroid_local to batch shape: [B, 1, 3]
        centroid_local_exp = centroid_local.expand(T, -1, -1)
        # Compute R @ centroid_local^T: first transpose centroid_local_exp to [B, 3, 1]
        T = centroid_global - torch.bmm(
            R, centroid_local_exp.transpose(-2, -1)
        ).transpose(-2, -1)
        T = T.squeeze(1)  # [B, 3]

        ee_pos = T
        # ee_pos = points_global[:,0,:]*0.5 + points_global[:,1,:]*0.5
        ee_rot = axis_angle_from_quat(matrix_to_quaternion(R))  # [B, 4]

        return ee_pos, ee_rot

    def points_to_gripper(self, gripper_points: torch.Tensor):
        global POINTS_FINGERS
        open_gripper_points = POINTS_FINGERS[-1]
        closed_gripper_points = POINTS_FINGERS[1]

        open_euler_dist = torch.norm(
            open_gripper_points[0] - open_gripper_points[1], dim=-1
        )
        closed_euler_dist = torch.norm(
            closed_gripper_points[0] - closed_gripper_points[1], dim=-1
        )

        current_euler_dist = torch.norm(
            gripper_points[:, 0] - gripper_points[:, 1], dim=-1
        )

        out_gripper = torch.ones(gripper_points.shape[0], device=gripper_points.device)
        out_gripper[
            abs(current_euler_dist - open_euler_dist)
            < abs(current_euler_dist - closed_euler_dist)
        ] = -1
        return out_gripper

    def ee_pose_to_3D_points(
        self,
        ee_pos: torch.Tensor,
        gripper: torch.Tensor,
        ee_rot: torch.Tensor = None,
        ee_quat: torch.Tensor = None,
    ) -> torch.Tensor:
        global POINTS_LOCAL, POINTS_FINGERS

        if ee_quat is None and ee_rot is None:
            raise ValueError("Either ee_quat or ee_rot must be provided.")
        if ee_quat is not None and ee_rot is not None:
            raise ValueError("Only one of ee_quat or ee_rot should be provided.")
        if ee_rot is not None:
            ee_quat = quat_from_axis_angle(ee_rot)

        num_samples = ee_pos.shape[0]
        num_total_points = POINTS_FINGERS.shape[1] + POINTS_LOCAL.shape[0]

        total_points = torch.empty(
            (num_samples, num_total_points, 3), device=ee_pos.device
        )
        total_points[:, : POINTS_LOCAL.shape[0], :] = POINTS_LOCAL.unsqueeze(0).expand(
            num_samples, -1, -1
        )
        total_points[:, POINTS_LOCAL.shape[0] :, :] = POINTS_FINGERS[
            gripper.cpu().squeeze().long()
        ]

        points_global: torch.Tensor = transform_points(total_points, ee_pos, ee_quat)

        return points_global

    def __call__(
        self,
        tensordict: TensorDict,
    ) -> TensorDict:
        return self.call_trajectory_rollout(tensordict[0]).unsqueeze(0)
