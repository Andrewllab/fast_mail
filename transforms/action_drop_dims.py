from __future__ import annotations

import dataclasses
import logging
from typing import Sequence

import torch
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import Transform, TransformConstraint

log = logging.getLogger(__name__)


class DropActionDims(Transform):
    # Cannot be applied to data from env, as the environment will presumably
    # not emit an observation with the ground truth action.
    constraints = [TransformConstraint.DATASET_ONLY]

    def __init__(self, specs: DataSpecs, indices_to_drop: Sequence[int]):
        if not indices_to_drop:
            raise ValueError("indices_to_drop cannot be empty")

        self.indices_to_drop = list(sorted(indices_to_drop))

        action_spec = specs.action
        action_dim = action_spec.action_dim
        new_action_spec = dataclasses.replace(
            action_spec, action_dim=action_dim - len(self.indices_to_drop)
        )

        self._specs = specs.replace(action=new_action_spec)

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # this transform can only be called during preprocessing
        assert "action" in tensordict
        action = tensordict["action"]

        indices = self.indices_to_drop
        indices = torch.as_tensor(indices, dtype=torch.long, device=tensordict.device)
        zeros = action[..., indices]
        if zeros.nonzero().numel() > 0:
            log.warning(
                f"Dropping action dimensions at indices {self.indices_to_drop}, "
                "but they are not all zero."
            )

        mask = torch.ones(action.shape[-1], dtype=torch.bool, device=tensordict.device)
        mask[indices] = False

        action = action[..., mask]

        # overwrite the old actions and ref actions
        tensordict["action"] = action
        # copy the original action in case it is modified in-place later
        tensordict["ref_action"] = action.clone()

        return tensordict
