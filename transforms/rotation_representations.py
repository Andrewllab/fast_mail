from __future__ import annotations

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import ReversibleTransform
from utils.math import normalize


class QuaternionRotations(ReversibleTransform):
    def __init__(
        self, specs: DataSpecs, remove_jumps: bool = True, mirror: bool = True
    ):
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

    def __call__(self, tensordict: TensorDict) -> TensorDict:
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
