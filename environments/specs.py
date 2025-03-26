from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class Spec:
    shape: tuple[int, ...]
    type: str
    camera_intrinsics: np.ndarray | None = None
    camera_extrinics: np.ndarray | None = None


@dataclass(frozen=True)
class DataSpecs:
    _obs: dict[str, Spec]
    action: Spec

    # def __init__(self, obs: Mapping[str, Spec], action: Spec):
    #     # frozen dataclass does not allow setting attributes after creation
    #     # wrap obs in MappingProxyType to make it immutable
    #     object.__setattr__(self, "obs", MappingProxyType(obs))
    #     object.__setattr__(self, "action", action)

    def copy(self) -> "DataSpecs":
        return DataSpecs(self.obs, self.action)

    @property
    def obs(self) -> dict[str, Spec]:
        # ensure that the obs attribute is never passed around and modified by accident
        return self._obs.copy()

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
    def embed_dim(self) -> int:
        try:
            embed_space = self.obs["obs_embed"]
        except KeyError:
            return 0

        return embed_space.shape[-1]

    @property
    def embed_seq_len(self) -> int:
        try:
            embed_space = self.obs["obs_embed"]
        except KeyError:
            return 0

        return embed_space.shape[0]

    @property
    def goal_seq_len(self) -> int:
        try:
            goal_space = self.obs["goal"]
        except KeyError:
            return 0

        return goal_space.shape[0]

    @property
    def goal_dim(self) -> int:
        # TODO: handle case where goal is not a vector
        try:
            goal_space = self.obs["goal"]
        except KeyError:
            return 0

        return goal_space.shape[-1]

    @property
    def goal_embed_seq_len(self) -> int:
        try:
            goal_embed_space = self.obs["goal_embed"]
        except KeyError:
            return 0

        return goal_embed_space.shape[0]

    @property
    def goal_embed_dim(self) -> int:
        try:
            goal_embed_space = self.obs["goal_embed"]
        except KeyError:
            return 0

        return goal_embed_space.shape[-1]
