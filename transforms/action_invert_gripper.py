from __future__ import annotations

import logging

import torch
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import ReversibleTransform, TransformConstraint

log = logging.getLogger(__name__)


class InvertGripperActions(ReversibleTransform):
    """Inverts gripper actions that represent a target width for the gripper.

    Args:
        specs (DataSpecs): data specs
    """

    def __init__(
        self,
        specs: DataSpecs,
    ):
        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        gripper_action = tensordict["action"][..., -1]

        tensordict["action"][..., -1] = -gripper_action

        if "_action" in tensordict:
            prev_action = tensordict["_action"][..., -1]
            tensordict["_action"][..., -1] = -prev_action

        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        # Inversion is symmetric, so we can just call the forward method again
        return self.__call__(tensordict)
