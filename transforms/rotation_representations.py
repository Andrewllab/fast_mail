from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import ReversibleTransform, TransformConstraint
from utils.math import axis_angle_from_quat, normalize, quat_from_angle_axis


class QuaternionRotations(ReversibleTransform):
    constraints = [
        # Cannot be applied to data from env, as the forward call modifies actions
        TransformConstraint.DATASET_ONLY,
        # Must be applied on trajectories to remove jumps and perform mirroring
        TransformConstraint.TRAJECTORY_ONLY,
    ]

    def __init__(
        self, specs: DataSpecs, remove_jumps: bool = True, mirror: bool = True
    ):
        action_spec = specs.action
        # This transform assumes 8D actions: 3 for position, 4 for quaternion,
        # 1 for gripper
        # TODO: more graceful handling of different action specs
        assert action_spec.action_dim == 8

        self._specs = specs
        self.remove_jumps = remove_jumps
        self.mirror = mirror

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(remove_jumps={self.remove_jumps}, mirror={self.mirror})"

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict | list[TensorDict]:
        if self.remove_jumps:
            quat = tensordict["action"][..., 3:7]
            quat = remove_jumps_from_quat_trajectory(quat)
            tensordict["action"][..., 3:7] = quat

        if self.mirror:
            mirrored = tensordict.clone(recurse=True)
            mirrored["action"][..., 3:7] *= -1.0
            return [tensordict, mirrored]
        else:
            return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:

        action = tensordict["action"]

        # normalize quaternions predicted by the model, which are likely not valid
        action[..., 3:7] = normalize(action[..., 3:7])

        return tensordict


def remove_jumps_from_quat_trajectory(quat: torch.Tensor) -> torch.Tensor:
    """Given a trajectory of quaternions, remove jumps by flipping quaternions where needed.

    Since a quaternion and its negation represent the same rotation, a trajectory of quaternions
    may contain jumps where a quaternion is suddenly replaced by its negation. This function detects
    such jumps and flips the quaternions after the jump to ensure a smooth trajectory.

    Args:
        quat (torch.Tensor): Tensor of shape (T, 4) representing a trajectory of quaternions.

    Returns:
        torch.Tensor: Tensor of the same shape as input, with jumps removed.
    """

    if quat.ndim != 2 or quat.shape[-1] != 4:
        raise ValueError("Input tensor must have shape (T, 4)")

    # create a tensor with both the quaternion and its negation
    quat_pos_and_neg = torch.stack([quat, -quat], dim=-2)

    # compare each quaternion to the following one and its negation and
    # compute the MSE
    mse = (quat_pos_and_neg[1:] - quat[:-1].unsqueeze(dim=-2)).pow(2).sum(dim=-1)

    # create a mask that is 0 if the original quaternion was closer, 1 if the
    # negated one was
    jump_locations = F.pad(torch.argmin(mse, dim=-1), (1, 0))

    # we have to swap all quaternions after a jump, but swaps cancel out so we
    # compute the cumulative sum of the mask
    swap_mask = (jump_locations.cumsum(dim=0) % 2).to(torch.bool)

    # swap the quaternions where needed
    quat[swap_mask] = -quat[swap_mask]

    return quat


class QuatToAxisAngleRotations(ReversibleTransform):
    # Cannot be applied to data from env, as the forward call modifies actions
    constraints = [TransformConstraint.DATASET_ONLY]

    def __init__(self, specs: DataSpecs):
        action_spec = specs.action
        # This transform assumes 8D actions: 3 for position, 4 for quaternion,
        # 1 for gripper
        # TODO: more graceful handling of different action specs
        assert action_spec.action_dim == 8

        new_action_dim = 7  # 3 for position + 3 for axis-angle + 1 for gripper
        new_action_spec = dataclasses.replace(action_spec, action_dim=new_action_dim)

        self._specs = specs.replace(action=new_action_spec)

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"]
        quat = action[..., 3:7]

        # reshape to 2D for conversion
        leading_dims = quat.shape[:-1]
        quat = quat.view(-1, 4)
        axis_angle = axis_angle_from_quat(quat)
        axis_angle = axis_angle.view(*leading_dims, 3)

        # replace quaternion in action with axis-angle
        new_action = torch.cat([action[..., :3], axis_angle, action[..., 7:]], dim=-1)
        tensordict["action"] = new_action

        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"]
        assert action.ndim == 3  # [B, T, 7]
        assert action.shape[-1] == 7
        B, T = action.shape[:2]
        axis_angle = action[..., 3:6]

        axis_angle = axis_angle.view(-1, 3)
        angle = axis_angle.norm(dim=-1)
        quat = quat_from_angle_axis(angle, axis_angle)
        quat = quat.view(B, T, 4)

        # replace axis-angle in action with quaternion
        new_action = torch.cat([action[..., :3], quat, action[..., 6:]], dim=-1)
        tensordict["action"] = new_action

        return tensordict
