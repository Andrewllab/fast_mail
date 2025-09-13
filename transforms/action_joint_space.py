from __future__ import annotations

import dataclasses
import logging

import torch
from tensordict import TensorDict

from environments.specs import DataSpecs, ActionSpec
from transforms.base_transform import Transform

log = logging.getLogger(__name__)


class JointSpaceActions(Transform):
    def __init__(self, specs: DataSpecs):

        action_spec = specs.action
        joint_pos_spec = specs.obs["target_joint_pos"]
        gripper_pos_spec = specs.obs["target_gripper_pos"]

        new_action_dim = joint_pos_spec.shape[-1] + gripper_pos_spec.shape[-1]
        new_action = dataclasses.replace(action_spec, action_dim=new_action_dim)
        specs = specs.replace(action=new_action)

        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:

        action = torch.cat(
            (
                tensordict["obs", "target_joint_pos"],  # shape: (T, 7)
                tensordict["obs", "target_gripper_pos"],  # shape: (T, 1)
            ),
            dim=-1,
        )

        # overwrite the old actions and ref actions
        tensordict["action"] = action
        tensordict["ref_action"] = action

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # This gets called when running with an environment, where the
        # preprocess transforms get rolled into the cpu_batch_transforms.
        # We don't need to do anything, since we can only run during
        # preprocessing.
        return tensordict
