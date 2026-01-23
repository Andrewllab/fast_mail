# absolute_ee_pose_to_five_points_transform.py
from __future__ import annotations

import dataclasses
import logging

import torch
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import ReversibleTransform
from utils.math import (
    axis_angle_from_quat,
    matrix_to_quaternion,
    quat_from_axis_angle,
    subtract_frame_transforms,
    transform_points,
)
from utils.nested import to_strided_tensor

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# 5-point action representation:
#   - First 3 points encode EE pose (translation + rotation) via rigid alignment (SVD).
#   - Last 2 points encode gripper state as symmetric fingertips along a known EE-local axis.
#
# Gripper convention (matches your prior points_to_gripper implementation):
#   gripper = -1  -> OPEN
#   gripper = +1  -> CLOSED
# --------------------------------------------------------------------------------------

# Three fixed points in EE local coordinates that define a stable frame for SVD pose recovery.
POINTS_LOCAL = torch.tensor(
    [
        [0.05, 0.0, -0.05],
        [-0.05, 0.0, -0.05],
        [0.0, 0.00, -0.1],
    ],
    dtype=torch.float32,
)  # (x, y, z) in EE local frame

# EE-local gripper geometry parameters.
# Axis along which the jaws open/close in the EE-local frame.
GRIPPER_AXIS_LOCAL = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)

# Center point (EE-local) around which the two fingertip points are placed.
# Tune this to where your jaws are relative to POINTS_LOCAL; doesn't have to be perfect.
GRIPPER_CENTER_LOCAL = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

# Widths (meters). Tune to your robot gripper.
W_OPEN = 0.10
W_CLOSED = 0.01


class AbsoluteEEPoseToFivePointsTransform(ReversibleTransform):
    """Convert action from (ee_pos, ee_rot, gripper) to 5x3 points and back.

    Action format in tensordict before transform:
        action[..., :3]    -> target ee_pos
        action[..., 3:6]   -> target ee_rot (axis-angle)
        action[..., 6:7]   -> gripper (scalar), typically in {-1, +1} (open/close)

    After transform:
        action -> (T, 5, 3) points in world frame (after base transform)
    """

    def __init__(
        self,
        specs: DataSpecs,
        ee_pose_key: str = "ee_pose",
        # Gripper decode options:
        binary_gripper: bool = True,
        use_hysteresis: bool = True,
        # If hysteresis is enabled, we switch:
        #   open  -> close when width < w_close_thresh
        #   close -> open  when width > w_open_thresh
        # with w_close_thresh < w_open_thresh.
        w_close_thresh: float | None = None,
        w_open_thresh: float | None = None,
        # Geometry params:
        w_open: float = W_OPEN,
        w_closed: float = W_CLOSED,
        gripper_axis_local: torch.Tensor = GRIPPER_AXIS_LOCAL,
        gripper_center_local: torch.Tensor = GRIPPER_CENTER_LOCAL,
    ):
        super().__init__()

        self._specs = specs
        self.ee_pose_key = ee_pose_key

        self.binary_gripper = binary_gripper
        self.use_hysteresis = use_hysteresis

        self.w_open = float(w_open)
        self.w_closed = float(w_closed)
        if not (self.w_open > self.w_closed):
            raise ValueError(
                f"Expected w_open > w_closed, got {self.w_open} <= {self.w_closed}"
            )

        # Default hysteresis thresholds (if not provided): midpoint +/- small margin
        # You should tune these from demo stats.
        mid = 0.5 * (self.w_open + self.w_closed)
        margin = 0.1 * (self.w_open - self.w_closed)
        self.w_close_thresh = (
            float(w_close_thresh) if w_close_thresh is not None else float(mid - margin)
        )
        self.w_open_thresh = (
            float(w_open_thresh) if w_open_thresh is not None else float(mid + margin)
        )
        if not (self.w_close_thresh < self.w_open_thresh):
            raise ValueError("Expected w_close_thresh < w_open_thresh for hysteresis.")

        # Store axis/center as buffers for easy device/dtype alignment
        self._gripper_axis_local = gripper_axis_local.to(torch.float32).clone()
        self._gripper_center_local = gripper_center_local.to(torch.float32).clone()

        # Update action spec to 5*3
        action_spec = specs.action
        new_action_spec = dataclasses.replace(action_spec, action_dim=5 * 3)
        self._specs = specs.replace(action=new_action_spec)

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    # ----------------------------------------------------------------------------------
    # Forward transforms
    # ----------------------------------------------------------------------------------

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        # Action from (pos, rot, gripper) -> 5 points (EE frame) -> world frame
        action_pos = tensordict["action"][..., :3]  # (T, 3)
        action_rot = tensordict["action"][..., 3:6]  # (T, 3) axis-angle
        gripper_action = tensordict["action"][..., 6:7]  # (T, 1)

        action_points = self.ee_pose_to_3D_points(
            ee_pos=action_pos, ee_rot=action_rot, gripper=gripper_action
        )  # (T, 5, 3) in world (EE pose is in world/base frame prior to base_pose transform below)

        # Transform action points to world frame via base_pose
        base_pose = tensordict["obs"]["base_pose"]  # (T, 7)
        base_pos = base_pose[..., :3]  # (T, 3)
        base_quat = base_pose[..., 3:]  # (T, 4)
        action_points = transform_points(action_points, base_pos, base_quat)

        # Compute current points for conditioning / observation
        current_pos = tensordict["obs"][self.ee_pose_key][..., :3]  # (T, 3)
        current_quat = tensordict["obs"][self.ee_pose_key][..., 3:]  # (T, 4)

        # Use previous action gripper as current gripper state (as you did)
        current_gripper = -torch.ones_like(gripper_action)  # default open
        current_gripper[1:] = tensordict["action"][:-1, 6:7]  # shift previous

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
            )  # (T,)
        else:
            gripper_state = tensordict["obs"]["_action"][:, 6].to(ee_pos.dtype)  # (T,)

        points = self.ee_pose_to_3D_points(
            ee_pos=ee_pos, ee_quat=ee_quat, gripper=gripper_state
        )  # (T, 5, 3)

        base_pose = tensordict["obs"]["base_pose"]  # (T, 7)
        base_pos = base_pose[..., :3]
        base_quat = base_pose[..., 3:]
        points = transform_points(points, base_pos, base_quat)

        tensordict["obs"]["gripper_points"] = {"points": points[0].to(torch.float32)}
        return tensordict

    # ----------------------------------------------------------------------------------
    # Reverse transform: 5 points -> (ee_pos, ee_rot, gripper)
    # ----------------------------------------------------------------------------------

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        points = tensordict["action"].float()  # expected (B, 5, T, 3) or nested

        if points.is_nested:
            if (torch.diff(points.offsets()) == 5).all():
                points = to_strided_tensor(points).unsqueeze(2)
            else:
                raise ValueError(
                    "Cannot reverse transform: action points tensor is jagged with unexpected offsets."
                )

        B, N, T, _ = points.shape
        if N != 5:
            raise ValueError(f"Expected 5 points, got {N}")

        # reshape to (B*T, 5, 3), operating in the same frame as stored in tensordict["action"]
        points = points.swapaxes(1, 2).reshape(-1, 5, 3)  # (B*T, 5, 3)

        # Inverse base transform: world -> base local (or whatever your original action frame is)
        base_pose = tensordict["obs"][
            "base_pose"
        ]  # (T, 7) or (B, T, 7)? you use [:,0] later.
        base_pos = base_pose[..., 0, :3]  # (T, 3)
        base_quat = base_pose[..., 0, 3:]  # (T, 4)

        inv_base_pos, inv_base_quat = subtract_frame_transforms(base_pos, base_quat)
        points = transform_points(points, inv_base_pos, inv_base_quat)

        # Pose from first 3 points (in "action frame")
        ee_pos, ee_rot = self.points_to_pose(
            points[:, :3]
        )  # (B*T, 3), (B*T, 3 axis-angle)

        # Gripper from last 2 points: project in EE local frame (robust)
        gripper = self.points_to_gripper(
            points[:, 3:5], ee_pos=ee_pos, ee_rot=ee_rot, B=B, T=T
        )  # (B*T,)

        ee_poses = torch.cat([ee_pos, ee_rot], dim=-1)  # (B*T, 6)
        ee_poses = ee_poses.view(B, T, 6)
        gripper = gripper.view(B, T)

        tensordict["action"] = torch.cat(
            [ee_poses, gripper.unsqueeze(-1)], dim=-1
        )  # (B, T, 7)
        return tensordict

    # ----------------------------------------------------------------------------------
    # Pose recovery from 3 points (SVD rigid alignment)
    # ----------------------------------------------------------------------------------

    @torch.no_grad()
    def points_to_pose(self, points_global: torch.Tensor):
        """
        Recover translation and rotation from 3 points.

        Args:
            points_global: [T, N=3, 3] (here T corresponds to B*T flattened)

        Returns:
            ee_pos: [T, 3]
            ee_rot: [T, 3] axis-angle
        """
        points_local = POINTS_LOCAL.to(points_global.device, points_global.dtype)

        Tn, N, _ = points_global.shape

        centroid_local = points_local.mean(dim=0, keepdim=True)  # [1, 3]
        X = points_local - centroid_local  # [N, 3]

        centroid_global = points_global.mean(dim=-2, keepdim=True)  # [Tn, 1, 3]
        Y = points_global - centroid_global  # [Tn, N, 3]

        X_T = X.t().unsqueeze(0).expand(Tn, 3, N)  # [Tn, 3, N]
        H = torch.bmm(X_T, Y)  # [Tn, 3, 3]

        U, S, Vh = torch.linalg.svd(H.to(torch.float32), full_matrices=False)
        V = Vh.transpose(-2, -1)

        R = torch.bmm(V, U.transpose(-2, -1))

        det_R = torch.det(R.to(torch.float32))
        # reflection correction
        for i in range(Tn):
            if det_R[i] < 0:
                V[i, :, -1] *= -1
        R = torch.bmm(V, U.transpose(-2, -1))

        centroid_local_exp = centroid_local.expand(Tn, -1, -1)  # [Tn, 1, 3]
        t = centroid_global - torch.bmm(
            R, centroid_local_exp.transpose(-2, -1)
        ).transpose(-2, -1)
        t = t.squeeze(1)  # [Tn, 3]

        ee_pos = t
        ee_rot = axis_angle_from_quat(
            matrix_to_quaternion(R)
        )  # [Tn, 3] via quat->axisangle
        return ee_pos, ee_rot

    # ----------------------------------------------------------------------------------
    # Gripper encode/decode
    # ----------------------------------------------------------------------------------

    def _gripper_width_from_action(self, gripper: torch.Tensor) -> torch.Tensor:
        """
        Map gripper action g in [-1,1] to a width in [w_closed, w_open]
        Convention:
            g = -1 => OPEN  => width = w_open
            g = +1 => CLOSED => width = w_closed
        """
        g = gripper
        if g.ndim > 1:
            g = g.squeeze(-1)
        g = g.to(torch.float32).clamp(-1.0, 1.0)

        # alpha = (1 - g)/2 gives:
        #   g=-1 => alpha=1 => open
        #   g=+1 => alpha=0 => closed
        alpha = 0.5 * (1.0 - g)
        width = self.w_closed + alpha * (self.w_open - self.w_closed)
        return width.to(gripper.dtype)

    def _action_from_gripper_width(self, width: torch.Tensor) -> torch.Tensor:
        """
        Invert width -> gripper action.
        width = w_open  => g = -1
        width = w_closed => g = +1
        """
        w = width.to(torch.float32)
        g = 1.0 - 2.0 * (w - self.w_closed) / (self.w_open - self.w_closed)
        return g.clamp(-1.0, 1.0).to(width.dtype)

    def _hysteresis_binarize(
        self, width_bt: torch.Tensor, init: float = -1.0
    ) -> torch.Tensor:
        """
        width_bt: (B, T)
        returns: (B, T) with values in {-1, +1} using hysteresis thresholds.
        """
        B, T = width_bt.shape
        out = torch.empty((B, T), device=width_bt.device, dtype=width_bt.dtype)
        state = torch.full(
            (B,), float(init), device=width_bt.device, dtype=width_bt.dtype
        )

        for t in range(T):
            w = width_bt[:, t]
            # closed -> open only if w > w_open_thresh
            state = torch.where(
                (state > 0) & (w > self.w_open_thresh), -torch.ones_like(state), state
            )
            # open -> close only if w < w_close_thresh
            state = torch.where(
                (state < 0) & (w < self.w_close_thresh), torch.ones_like(state), state
            )
            out[:, t] = state
        return out

    def points_to_gripper(
        self,
        gripper_points_world: torch.Tensor,
        *,
        ee_pos: torch.Tensor,
        ee_rot: torch.Tensor,
        B: int,
        T: int,
    ) -> torch.Tensor:
        """
        Decode gripper from the last two points by:
          1) transforming the two points into EE local frame
          2) projecting their delta onto the known local gripper axis
          3) mapping width -> gripper action (continuous or binary)

        Args:
            gripper_points_world: (B*T, 2, 3) in the same frame as ee_pos/ee_rot
            ee_pos: (B*T, 3)
            ee_rot: (B*T, 3) axis-angle
        Returns:
            gripper: (B*T,) in {-1,+1} if binary_gripper else in [-1,1]
        """
        device = gripper_points_world.device
        dtype = gripper_points_world.dtype

        ee_quat = quat_from_axis_angle(ee_rot.to(dtype))  # (B*T, 4)

        # world -> EE local
        inv_pos, inv_quat = subtract_frame_transforms(ee_pos, ee_quat)
        fingers_local = transform_points(
            gripper_points_world, inv_pos, inv_quat
        )  # (B*T, 2, 3)

        delta = fingers_local[:, 1, :] - fingers_local[:, 0, :]  # (B*T, 3)

        axis = self._gripper_axis_local.to(device=device, dtype=dtype)
        # Width is the magnitude of delta along the opening axis
        width = torch.abs((delta * axis[None, :]).sum(dim=-1))  # (B*T,)

        if not self.binary_gripper:
            return self._action_from_gripper_width(width)

        # binary decode
        if self.use_hysteresis:
            width_bt = width.view(B, T)
            g_bt = self._hysteresis_binarize(width_bt, init=-1.0)
            return g_bt.reshape(-1)

        # simple midpoint threshold (no temporal state)
        mid = 0.5 * (self.w_open + self.w_closed)
        # width closer to open -> g=-1, width closer to closed -> g=+1
        g = torch.where(width >= mid, -torch.ones_like(width), torch.ones_like(width))
        return g

    def ee_pose_to_3D_points(
        self,
        ee_pos: torch.Tensor,
        gripper: torch.Tensor,
        ee_rot: torch.Tensor | None = None,
        ee_quat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Build 5 points in world frame from EE pose and gripper action.

        First 3 points are fixed in EE local frame: POINTS_LOCAL
        Last 2 points are symmetric along GRIPPER_AXIS_LOCAL with width determined by gripper action.
        """
        if ee_quat is None and ee_rot is None:
            raise ValueError("Either ee_quat or ee_rot must be provided.")
        if ee_quat is not None and ee_rot is not None:
            raise ValueError("Only one of ee_quat or ee_rot should be provided.")
        if ee_rot is not None:
            ee_quat = quat_from_axis_angle(ee_rot)

        num_samples = ee_pos.shape[0]

        # Align buffers to current device/dtype
        pts_local = POINTS_LOCAL.to(device=ee_pos.device, dtype=ee_pos.dtype)
        axis = self._gripper_axis_local.to(device=ee_pos.device, dtype=ee_pos.dtype)
        center = self._gripper_center_local.to(device=ee_pos.device, dtype=ee_pos.dtype)

        # Map gripper action -> width
        width = self._gripper_width_from_action(gripper).to(
            device=ee_pos.device, dtype=ee_pos.dtype
        )  # (N,)

        # Construct fingertip points in EE local frame
        pL_local = center[None, :] - 0.5 * width[:, None] * axis[None, :]
        pR_local = center[None, :] + 0.5 * width[:, None] * axis[None, :]

        # Pack full 5 points in EE local frame
        total_points_local = torch.empty(
            (num_samples, 5, 3), device=ee_pos.device, dtype=ee_pos.dtype
        )
        total_points_local[:, :3, :] = pts_local.unsqueeze(0).expand(
            num_samples, -1, -1
        )
        total_points_local[:, 3, :] = pL_local
        total_points_local[:, 4, :] = pR_local

        # EE local -> world
        points_global = transform_points(total_points_local, ee_pos, ee_quat)
        return points_global

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Keep your existing behavior
        return self.call_trajectory_rollout(tensordict[0]).unsqueeze(0)
