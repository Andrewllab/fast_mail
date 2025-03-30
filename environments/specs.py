from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from frozendict import frozendict

log = logging.getLogger(__name__)


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
    a_mean: torch.Tensor | None = None
    a_var: torch.Tensor | None = None
    a_min: torch.Tensor | None = None
    a_max: torch.Tensor | None = None
    n_actions: int = 0

    def __init__(
        self, shape: tuple[int, ...], type: str, actions: torch.Tensor | None = None
    ):
        super().__init__(shape, type)

        mean = actions.mean(0) if actions is not None else None
        var = actions.var(0) if actions is not None else None
        min = actions.min(0).values if actions is not None else None
        max = actions.max(0).values if actions is not None else None
        n_actions = actions.shape[0] if actions is not None else 0
        self._set_stats(mean, var, min, max, n_actions)

    def _set_stats(self, mean, var, min, max, n_actions):
        # frozen dataclass does not allow setting attributes after creation
        object.__setattr__(self, "a_mean", mean)
        object.__setattr__(self, "a_var", var)
        object.__setattr__(self, "a_min", min)
        object.__setattr__(self, "a_max", max)
        object.__setattr__(self, "n_actions", n_actions)

    @property
    def a_std(self) -> torch.Tensor | None:
        return self.a_var.sqrt() if self.a_var is not None else None

    def update_stats(self, actions: torch.Tensor) -> None:
        m = len(actions)
        if m == 0:
            return  # No update needed if the batch is empty

        if (n := self.n_actions) == 0:
            assert all(
                v is None for v in (self.a_mean, self.a_var, self.a_min, self.a_max)
            )
            self._set_stats(
                actions.mean(0),
                actions.var(0),
                actions.min(0).values,
                actions.max(0).values,
                m,
            )
            return

        mean, var, min, max = self.a_mean, self.a_var, self.a_min, self.a_max
        assert all(v is not None for v in (mean, var, min, max))

        new_mean = actions.mean(0)
        new_var = actions.var(0, correction=0)

        total_n = n + m
        delta_mean = new_mean - mean
        total_mean = mean + m * delta_mean / total_n
        total_var = (
            n * var + m * new_var + (n * m / total_n) * delta_mean**2
        ) / total_n

        new_min, new_max = actions.min(0).values, actions.max(0).values

        self._set_stats(
            total_mean,
            total_var,
            torch.minimum(min, new_min),
            torch.maximum(max, new_max),
            total_n,
        )


class DataSpecs:
    def __init__(
        self,
        obs: Mapping[str, Spec],
        action: ActionSpec,
        goal: Mapping[str, Spec] | None = None,
        lengths: Sequence[int] | None = None,
    ):
        self._obs = frozendict(obs)
        self._action = action
        self._goal = frozendict(goal) if goal is not None else None
        self._lengths = list(lengths) if lengths is not None else []

    def __repr__(self) -> str:
        args = (
            dict(self._obs),
            self._action,
            dict(self._goal) if self._goal is not None else None,
            self._lengths,
        )
        return "DataSpecs(obs={}, action={}, goal={}, lengths={})".format(*args)

    def replace(self, **kwargs) -> DataSpecs:
        default_kwargs = {
            "obs": self._obs,
            "action": self._action,
            "goal": self._goal,
            "lengths": self._lengths,
        }
        default_kwargs.update(kwargs)

        return DataSpecs(**default_kwargs)

    @property
    def obs(self) -> frozendict[str, Spec]:
        return self._obs

    @property
    def action(self) -> ActionSpec:
        return self._action

    @property
    def goal(self) -> frozendict[str, Spec] | None:
        return self._goal

    @property
    def lengths(self) -> list[int]:
        # copy the list to avoid mutation
        return list(self._lengths)

    def append_length(self, length: int) -> None:
        self._lengths.append(length)

    def extend_lengths(self, lengths: Sequence[int]) -> None:
        self._lengths.extend(lengths)

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


def save_specs(specs: DataSpecs, path: os.PathLike) -> None:
    with open(path, "wb") as f:
        pickle.dump(specs, f)

    log.debug(f"Specs saved to {path}")


def load_specs(path: os.PathLike) -> DataSpecs:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Specs file {path} does not exist")

    with open(path, "rb") as f:
        specs = pickle.load(f)

    log.debug(f"Specs loaded from {path}")
    return specs
