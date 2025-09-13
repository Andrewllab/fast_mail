from __future__ import annotations

import logging

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, NormalizingTransform

log = logging.getLogger(__name__)


class ActionNormMaxMagnitude(NormalizingTransform, nn.Module):
    max_actions: torch.Tensor

    def __init__(self, specs: DataSpecs):
        super().__init__()

        self._specs = specs

        self.register_buffer("max_actions", torch.zeros(specs.action_dim))

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        # update action statistics
        actions = tensordict["action"]

        # flatten all but the last dimension
        actions = actions.flatten(start_dim=0, end_dim=-2)

        max_actions = actions.abs().max(dim=0).values
        self.max_actions[...] = torch.maximum(self.max_actions, max_actions)

        return tensordict

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [KeyMapping(in_keys="action", out_keys="action")]

    def _call_one(self, action: Tensor) -> Tensor:
        if (self.max_actions == 0.0).any():
            log.warning(
                "Some dimensions of the action space have a max magnitude of zero. This will cause NaN values after action normalization."
            )

        # normalize actions
        action = action / self.max_actions
        return action

    def _reverse_one(self, action: Tensor) -> Tensor:
        if (self.max_actions == 0.0).any():
            log.warning(
                "Some dimensions of the action space have a max magnitude of zero. This will cause zero values after action unnormalization."
            )

        # clamp action to [-1, 1]
        action = action.clamp(-1, 1)

        # unnormalize actions
        action = action * self.max_actions
        return action


class ActionNormMinMax(NormalizingTransform, nn.Module):
    max_actions: torch.Tensor
    min_actions: torch.Tensor
    range_actions: torch.Tensor

    def __init__(self, specs: DataSpecs):
        super().__init__()

        self._specs = specs

        self.register_buffer("max_actions", torch.zeros(specs.action_dim))
        self.register_buffer("min_actions", torch.zeros(specs.action_dim))
        self.register_buffer("range_actions", torch.zeros(specs.action_dim))

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        # update action statistics
        actions = tensordict["action"]

        # flatten all but the last dimension
        actions = actions.flatten(start_dim=0, end_dim=-2)

        max_actions = actions.max(dim=0).values
        self.max_actions[...] = torch.maximum(self.max_actions, max_actions)

        min_actions = actions.min(dim=0).values
        self.min_actions[...] = torch.minimum(self.min_actions, min_actions)

        self.range_actions[...] = self.max_actions - self.min_actions

        return tensordict

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [KeyMapping(in_keys="action", out_keys="action")]

    def _call_one(self, action: Tensor) -> Tensor:
        if (self.range_actions == 0.0).any():
            log.warning(
                "Some dimensions of the action space have a range of zero. This will cause NaN values after action normalization."
            )

        # normalize actions
        action = (action - self.min_actions) / self.range_actions * 2 - 1
        return action

    def _reverse_one(self, action: Tensor) -> Tensor:
        if (self.range_actions == 0.0).any():
            log.warning(
                "Some dimensions of the action space have a range of zero. This will cause zero values after action unnormalization."
            )

        # clamp action to [-1, 1]
        action = action.clamp(-1, 1)

        # unnormalize actions
        action = (action + 1) / 2 * self.range_actions + self.min_actions
        return action
