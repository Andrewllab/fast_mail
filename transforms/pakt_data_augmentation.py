from __future__ import annotations

from typing import Sequence, Union

import torch
from tensordict import TensorDict
from torch_geometric.data import Data

from environments.specs import DataSpecs, ObsSpec
from transforms.base_transform import ReversibleTransform, Transform
from transforms.pointmap_random_transform import BaseRandomlyTransform
from utils.math import make_pose, quat_mul, quaternion_to_matrix, transform_pointcloud


def get_random_rotation_axes(
    axes: Union[str, Sequence[str]] = "x",
    device: Union[str, torch.device] = "cpu",
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Generate random rotation quaternions *only* about the specified axis or axes.

    Args:
        axes:   One of "x", "y", "z", or a list/tuple of two of these.
        device: torch device
        batch_size: how many quaternions to sample.

    Returns:
        Tensor of shape (batch_size, 4) of unit quaternions [w, x, y, z].
    """
    # normalize input to a list
    if isinstance(axes, str):
        axes = [axes]
    assert 1 <= len(axes) <= 2, "axes must be 'x','y','z', or a pair of them"

    # map axis names to unit vectors
    axis_map = {
        "x": torch.tensor([1.0, 0.0, 0.0], device=device),
        "y": torch.tensor([0.0, 1.0, 0.0], device=device),
        "z": torch.tensor([0.0, 0.0, 1.0], device=device),
    }

    # sample uniform angles in [0,2π) for each requested axis
    thetas = torch.rand(batch_size, len(axes), device=device) * 2 * torch.pi

    # build one quaternion per axis
    qs = []
    for i, ax in enumerate(axes):
        u = axis_map[ax].unsqueeze(0).expand(batch_size, -1)  # (B,3)
        half = thetas[:, i] * 0.5  # (B,)
        w = torch.cos(half).unsqueeze(-1)  # (B,1)
        xyz = torch.sin(half).unsqueeze(-1) * u  # (B,3)
        qs.append(torch.cat([w, xyz], dim=-1))  # (B,4)

    # if only one axis, that's our rotation
    if len(qs) == 1:
        return qs[0]

    # if two axes, compose: first rotate about axes[0], then about axes[1]
    # using your provided quat_mul
    # note: quat_mul(q1, q2) means “apply q2, then q1”
    return quat_mul(qs[1], qs[0])


class RandomRotation(ReversibleTransform):
    """Apply a random rotation to specified obs pointclouds and action points."""

    def __init__(
        self,
        specs: DataSpecs,
        rotation_axes: Union[str, Sequence[str]] = "x",
        obs_keys: Sequence[str] = (
            "target_points",
            "tool_points",
            "gripper_points",
            "des_gripper_points",
        ),
        action_keys: Sequence[str] = ("action",),
    ):
        obs_specs = dict(specs.obs)
        obs_specs["random_rotation_tf"] = ObsSpec((3, 3))
        obs_specs["random_rotation_quat"] = ObsSpec((4,))
        self._specs = specs.replace(obs=obs_specs)

        self.rotation_axes = rotation_axes
        self.obs_keys = list(obs_keys)
        self.action_keys = list(action_keys)

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    @torch.no_grad()
    def _rotate_batched_rowvec(self, x: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Rotate row-vectors x using per-batch rotation matrices.

        Supports:
        - Dense tensors: x shape (B, ..., 3)
        - Nested jagged tensors: x is torch.nested (layout=jagged), with batch dim first.
            Each element x[b] must be a dense tensor shaped (..., 3).

        Convention:
        row-vectors: x' = x @ R^T
        """
        if R.ndim != 3 or R.shape[-2:] != (3, 3):
            raise ValueError(f"Expected R of shape (B,3,3), got {tuple(R.shape)}")
        B = R.shape[0]
        Rt = R.transpose(-1, -2)

        # Nested jagged: rotate each batch element with its own matrix
        if getattr(x, "is_nested", False):
            if x.size(0) != B:
                raise ValueError(f"Batch mismatch: x has B={x.size(0)} but R has B={B}")

            out = []
            for b in range(B):
                xb = x[b]  # dense (..., 3)
                if xb.numel() == 0:
                    out.append(xb)
                    continue
                if xb.shape[-1] != 3:
                    raise ValueError(
                        f"Expected last dim 3 for nested element {b}, got {xb.shape[-1]}"
                    )
                out.append(torch.einsum("...j,ij->...i", xb, Rt[b]))

            return torch.nested.as_nested_tensor(out, layout=torch.jagged)

        # Dense
        if x.ndim < 2:
            raise ValueError(
                f"Expected x to have at least 2 dims (B,...,3), got {x.ndim}"
            )
        if x.shape[0] != B:
            raise ValueError(f"Batch mismatch: x has B={x.shape[0]} but R has B={B}")
        if x.shape[-1] != 3:
            raise ValueError(f"Expected last dim 3 for points, got {x.shape[-1]}")

        return torch.einsum("b...j,bij->b...i", x, Rt)

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        device = tensordict.device or "cpu"
        B = tensordict.batch_size[0]

        quat = get_random_rotation_axes(
            self.rotation_axes, device=device, batch_size=B
        )  # [w,x,y,z]
        R = quaternion_to_matrix(quat)  # (B,3,3) in your utils' convention

        # Rotate obs point clouds (only their "points" field)
        for key in self.obs_keys:
            pts = tensordict["obs", key, "points"]
            tensordict["obs", key, "points"] = self._rotate_batched_rowvec(pts, R)

        # Rotate action points
        for key in self.action_keys:
            tensordict[key] = self._rotate_batched_rowvec(tensordict[key], R)

        tensordict["obs"]["random_rotation_tf"] = R
        tensordict["obs"]["random_rotation_quat"] = quat
        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        if "random_rotation_tf" not in tensordict["obs"]:
            return tensordict

        R = tensordict["obs"]["random_rotation_tf"]  # (B,3,3)

        # Forward used: x' = x @ R^T
        # So reverse must use: x = x' @ R
        # Our helper always does x @ (matrix)^T, so pass (R^T) to get x @ R:
        R_for_reverse = R.transpose(-1, -2)

        for key in self.action_keys:
            tensordict[key] = self._rotate_batched_rowvec(
                tensordict[key], R_for_reverse
            )

        return tensordict
