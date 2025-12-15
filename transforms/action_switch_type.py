from __future__ import annotations

import dataclasses
import logging

import torch
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import Transform, TransformConstraint

log = logging.getLogger(__name__)


class SwitchActionType(Transform):
    # Cannot be applied to data from env, as the environment will presumably
    # not emit an observation with the ground truth action.
    constraints = [TransformConstraint.DATASET_ONLY]

    def __init__(self, specs: DataSpecs, obs_key: str):
        if obs_key not in specs.obs:
            raise ValueError(f"Key '{obs_key}' not found in obs specs.")

        action_spec = specs.action
        # TODO: more graceful handling of different action specs
        obs_spec_for_action = specs.obs[obs_key]
        new_action_dim = obs_spec_for_action.shape[-1] + 1  # +1 for gripper command
        new_action_spec = dataclasses.replace(action_spec, action_dim=new_action_dim)

        self._specs = specs.replace(action=new_action_spec)
        self.obs_key = obs_key

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # this transform can only be called during preprocessing
        assert "action" in tensordict

        action = torch.cat(
            (
                tensordict["obs", self.obs_key],  # shape: (T, N)
                tensordict["action"][..., -1:],  # shape: (T, 1)
            ),
            dim=-1,
        )

        # overwrite the old actions and ref actions
        tensordict["action"] = action
        # copy the original action in case it is modified in-place later
        tensordict["ref_action"] = action.clone()

        return tensordict
