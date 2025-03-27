from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from frozendict import frozendict


@dataclass(frozen=True)
class Spec:
    shape: tuple[int, ...]
    type: str


@dataclass(frozen=True)
class CameraSpec(Spec):
    camera_intrinsics: np.ndarray | None = None
    camera_extrinics: np.ndarray | None = None


@dataclass(frozen=True)
class ActionSpec(Spec):
    a_mean: torch.Tensor
    a_std: torch.Tensor
    a_min: torch.Tensor
    a_max: torch.Tensor


@dataclass(frozen=True)
class DataSpecs:
    obs: frozendict[str, Spec]
    action: ActionSpec
    goal: frozendict[str, Spec] | None = None
    goal_embed: Spec | None = None

    def __init__(
        self,
        obs: Mapping[str, Spec],
        action: ActionSpec,
        goal: Mapping[str, Spec] | None = None,
        goal_embed: Spec | None = None,
    ):
        # frozen dataclass does not allow setting attributes after creation
        object.__setattr__(self, "obs", frozendict(obs))
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "goal", frozendict(goal) if goal is not None else None)
        object.__setattr__(self, "goal_embed", goal_embed)

    @property
    def state_dim(self) -> int:
        try:
            robot_state_space = self.obs["robot_state"]
        except KeyError:
            return 0

        return robot_state_space.shape[-1]

    @property
    def action_seq_len(self) -> int:
        assert len(self.action.shape) == 2
        return self.action.shape[0]

    @property
    def action_dim(self) -> int:
        assert len(self.action.shape) == 2
        return self.action.shape[1]

    @property
    def obs_embed_dim(self) -> int:
        try:
            embed_space = self.obs["obs_embed"]
        except KeyError:
            return 0

        return embed_space.shape[-1]

    @property
    def obs_embed_seq_len(self) -> int:
        try:
            embed_space = self.obs["obs_embed"]
        except KeyError:
            return 0

        return embed_space.shape[0]

    @property
    def goal_embed_seq_len(self) -> int:
        if self.goal_embed is None:
            return 0

        return self.goal_embed.shape[0]

    @property
    def goal_embed_dim(self) -> int:
        if self.goal_embed is None:
            return 0

        return self.goal_embed.shape[-1]
