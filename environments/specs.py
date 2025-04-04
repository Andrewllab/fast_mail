from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from frozendict import frozendict

from utils.math import invert_intrinsics

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Spec:
    shape: tuple[int, ...]
    type: str


@dataclass
class PinholeCameraIntrinsic:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __init__(
        self, width: int, height: int, fx: float, fy: float, cx: float, cy: float
    ):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        self._intrinsic_matrix = torch.tensor(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ]
        )

        self._inverse_matrix = invert_intrinsics(self._intrinsic_matrix)

    @property
    def intrinsic_matrix(self) -> torch.Tensor:
        return self._intrinsic_matrix

    @property
    def inverse_matrix(self) -> torch.Tensor:
        return self._inverse_matrix

    def resize(self, new_shape: tuple[int, int]) -> PinholeCameraIntrinsic:
        new_width, new_height = new_shape
        scale_x = new_width / self.width
        scale_y = new_height / self.height

        return PinholeCameraIntrinsic(
            width=new_width,
            height=new_height,
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
        )


@dataclass(frozen=True)
class CameraSpec(Spec):
    intrinsics: PinholeCameraIntrinsic | None = None
    extrinsics: torch.Tensor | None = None
    # the key (within the obs dict) with the dynamic pose information for the camera
    # e.g. a wrist camera would have extrinsics relative to the end effector pose
    dynamic_pose_obs_key: str | tuple[str, ...] | None = None
    type: str = "rgb"


@dataclass(frozen=True)
class DepthCameraSpec(CameraSpec):
    # the key (within the obs dict) where the corresponding rgb image is
    rgb_obs_key: str | tuple[str, ...] | None = None
    # orthogonal or perspective depth measurement
    orthogonal: bool = True
    type: str = "depth"


@dataclass(frozen=True)
class PointCloudSpec(Spec):
    type: str = "pointcloud"


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
        # TODO: replace with gymnasium.wrappers.utils.RunningMeanStd
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DataSpecs):
            return False

        return (
            self._obs == other._obs
            and self._action == other._action
            and self._goal == other._goal
            and self._lengths == other._lengths
        )

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
