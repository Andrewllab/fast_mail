from __future__ import annotations

from typing import Sequence, Union

import torch
from tensordict import TensorDict
from torch_geometric.data import Data

from environments.specs import DataSpecs, ObsSpec
from transforms.base_transform import ReversibleTransform, Transform
from transforms.pointmap_random_transform import BaseRandomlyTransform
from utils.math import make_pose, quaternion_to_matrix, transform_pointcloud


class RandomRotation(Transform):
    """Apply a random translation and rotation to the entire pointcloud
    trajectory during preprocessing, which is then constant throughout
    training.

    Args:
        specs (DataSpecs): The data specifications.
    """

    def __init__(
        self,
        specs: DataSpecs,
    ):
        obs_specs = dict(specs.obs)
        obs_specs["random_rotation_tf"] = ObsSpec((4, 4))
        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def random_rotation(self, device: Union[str, torch.device] = "cpu") -> torch.Tensor:
        """
        Generate a random rotation quaternion as a PyTorch tensor.

        Returns:
            torch.Tensor: Random rotation quaternion (4,).
        """
        # Generate a random quaternion
        # Source https://imois.in/posts/random-vectors-and-rotations-in-3d/#Quaternions
        # (III.6 - UNIFORM RANDOM ROTATIONS, Shoemake, 1992)
        q = torch.normal(mean=0, std=1, size=(1, 4), device=device)
        q = q / torch.norm(q, dim=1, keepdim=True)

        return q

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        device = tensordict.device or "cpu"

        rotation = self.random_rotation(device)
        transform = make_pose(
            torch.zeros((1, 3), device=device), quaternion_to_matrix(rotation)
        )
        transform = transform.squeeze(0)

        for key in ["target_points", "tool_points", "gripper_points"]:
            data: Data = tensordict["obs", key]
            data.pos = transform_pointcloud(data.pos, transform)
        for key in ["action", "ref_action"]:
            tensordict[key] = transform_pointcloud(tensordict[key], transform)

        tensordict["obs"]["random_rotation_tf"] = transform.unsqueeze(0)

        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        transform = tensordict["obs"]["random_rotation_tf"]
        inv_transform = torch.inverse(transform)

        tensordict["action"] = transform_pointcloud(tensordict["action"], inv_transform)
        tensordict["ref_action"] = transform_pointcloud(
            tensordict["ref_action"], inv_transform
        )

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Do nothing if not called during preprocessing
        return tensordict
