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
    # the key describing the pose information that the camera_extrinsics are relative to
    # e.g. a wrist camera would have extrinsics relative to the end effector pose
    mount_point: str | tuple[str, ...] | None = None


@dataclass(frozen=True)
class ActionSpec(Spec):
    a_mean: torch.Tensor | None
    a_std: torch.Tensor | None
    a_min: torch.Tensor | None
    a_max: torch.Tensor | None

    def __init__(
        self, shape: tuple[int, ...], type: str, all_actions: torch.Tensor | None = None
    ):
        super().__init__(shape, type)

        # frozen dataclass does not allow setting attributes after creation
        mean = all_actions.mean(0) if all_actions is not None else None
        std = all_actions.std(0) if all_actions is not None else None
        min = all_actions.min(0).values if all_actions is not None else None
        max = all_actions.max(0).values if all_actions is not None else None
        object.__setattr__(self, "a_mean", mean)
        object.__setattr__(self, "a_std", std)
        object.__setattr__(self, "a_min", min)
        object.__setattr__(self, "a_max", max)


@dataclass(frozen=True)
class DataSpecs:
    obs: frozendict[str, Spec]
    action: ActionSpec
    goal: frozendict[str, Spec] | None = None

    def __init__(
        self,
        obs: Mapping[str, Spec],
        action: ActionSpec,
        goal: Mapping[str, Spec] | None = None,
    ):
        # frozen dataclass does not allow setting attributes after creation
        object.__setattr__(self, "obs", frozendict(obs))
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "goal", frozendict(goal) if goal is not None else None)

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
            embed_space = self.obs["embed"]
        except KeyError:
            return 0

        return embed_space.shape[-1]

    @property
    def obs_embed_seq_len(self) -> int:
        try:
            embed_space = self.obs["embed"]
        except KeyError:
            return 0

        return embed_space.shape[0]

    @property
    def goal_embed_seq_len(self) -> int:
        if self.goal is None:
            return 0

        try:
            return self.goal["embed"].shape[0]
        except KeyError:
            return 0

    @property
    def goal_embed_dim(self) -> int:
        if self.goal is None:
            return 0

        try:
            return self.goal["embed"].shape[-1]
        except KeyError:
            return 0
