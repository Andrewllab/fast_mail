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
    """Apply a random translation and rotation to the entire pointcloud
    trajectory during preprocessing, which is then constant throughout
    training.

    Args:
        specs (DataSpecs): The data specifications.
    """

    def __init__(
        self,
        specs: DataSpecs,
        rotation_axes: Union[str, Sequence[str]] = "x",
    ):
        obs_specs = dict(specs.obs)
        obs_specs["random_rotation_tf"] = ObsSpec((4, 4))
        self._specs = specs.replace(obs=obs_specs)

        self.rotation_axes = rotation_axes

        self.obs_keys = ["target_points", "tool_points", "gripper_points"]
        self.action_keys = [
            "action",
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        device = tensordict.device or "cpu"
        B = tensordict.batch_size[0]

        rotation = get_random_rotation_axes(
            self.rotation_axes, device=device, batch_size=B
        )
        transform = quaternion_to_matrix(rotation)  # (B,4,4)

        for key in self.obs_keys:
            tensordict["obs", key, "points"] = tensordict["obs", key, "points"].matmul(
                transform
            )
        for key in self.action_keys:
            tensordict[key] = tensordict[key].matmul(transform)

        tensordict["obs"]["random_rotation_tf"] = transform

        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        transform = tensordict["obs"]["random_rotation_tf"]
        inv_transform = torch.inverse(transform)

        for key in self.action_keys:
            tensordict[key] = tensordict[key].matmul(inv_transform)

        return tensordict
