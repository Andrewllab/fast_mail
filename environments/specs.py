from __future__ import annotations

import logging
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Literal, Mapping, Sequence

import numpy as np
import torch
from frozendict import frozendict

from utils.math import invert_intrinsics

log = logging.getLogger(__name__)


@dataclass
class PinholeCameraIntrinsic:
    height: int
    width: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __init__(
        self, height: int, width: int, fx: float, fy: float, cx: float, cy: float
    ):
        self.height = height
        self.width = width
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

    @classmethod
    def from_intrinsic_matrix(
        cls, intrinsic_matrix: torch.Tensor, height: int, width: int
    ) -> PinholeCameraIntrinsic:
        fx = intrinsic_matrix[0, 0].item()
        fy = intrinsic_matrix[1, 1].item()
        cx = intrinsic_matrix[0, 2].item()
        cy = intrinsic_matrix[1, 2].item()

        return cls(height=height, width=width, fx=fx, fy=fy, cx=cx, cy=cy)

    @property
    def intrinsic_matrix(self) -> torch.Tensor:
        return self._intrinsic_matrix

    @property
    def inverse_matrix(self) -> torch.Tensor:
        return self._inverse_matrix

    def resize(self, new_shape: tuple[int, int]) -> PinholeCameraIntrinsic:
        new_height, new_width = new_shape
        scale_y = new_height / self.height
        scale_x = new_width / self.width

        return PinholeCameraIntrinsic(
            height=new_height,
            width=new_width,
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=self.cx * scale_x,
            cy=self.cy * scale_y,
        )

    def center_crop(self, new_shape: tuple[int, int]) -> PinholeCameraIntrinsic:
        new_height, new_width = new_shape
        crop_y = (self.height - new_height) / 2
        crop_x = (self.width - new_width) / 2

        return PinholeCameraIntrinsic(
            height=new_height,
            width=new_width,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx - crop_x,
            cy=self.cy - crop_y,
        )


class Spec(ABC):
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the data. The first dimension is always the batch dimension."""
        pass


@dataclass(frozen=True)
class ObsSpec(Spec):
    elem_shape: tuple[int, ...]
    time: int | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        if self.time is None:
            return self.elem_shape
        else:
            return (self.time, *self.elem_shape)


RGBChannelOrderType = Literal["HWC", "CHW"]
ChannelOrderType = Literal["HWC", "CHW", "HW"]


@dataclass(frozen=True)
class ImageStream(ABC):
    """Base class for all image streams. By convention, rgb and monocolor
    intensity streams are dtype uint8, while depth streams are float32.
    """

    height: int
    width: int
    channels: int | None = None
    channel_order: ChannelOrderType = "HW"
    time: int | None = None
    intrinsics: PinholeCameraIntrinsic | None = None

    @property
    def height_width(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def image_shape(self) -> tuple[int, int] | tuple[int, int, int]:
        if self.channel_order == "HWC":
            assert self.channels is not None
            return self.height, self.width, self.channels
        elif self.channel_order == "CHW":
            assert self.channels is not None
            return self.channels, self.height, self.width
        elif self.channel_order == "HW":
            return self.height, self.width
        else:
            raise ValueError(f"Unknown channel order: {self.channel_order}")

    @property
    def n_image_dims(self) -> int:
        if self.channels is None:
            return 2
        return 3

    @property
    def shape(self) -> tuple[int, ...]:
        if self.time is None:
            return self.image_shape
        else:
            return (self.time, *self.image_shape)

    def reorder_channels(self, channel_order: ChannelOrderType) -> ImageStream:
        if self.channels is None or self.channel_order == "HW":
            return self

        if channel_order == self.channel_order:
            return self

        return replace(self, channel_order=channel_order)

    def center_crop(self, new_shape: tuple[int, int]) -> ImageStream:
        """Crop the image to the specified size."""
        new_height, new_width = new_shape
        new_intrinsics = (
            self.intrinsics.center_crop(new_shape)
            if self.intrinsics is not None
            else None
        )
        return replace(
            self, height=new_height, width=new_width, intrinsics=new_intrinsics
        )

    def resize(self, new_shape: tuple[int, int]) -> ImageStream:
        """Resize the image to the specified size."""
        new_height, new_width = new_shape
        new_intrinsics = (
            self.intrinsics.resize(new_shape) if self.intrinsics is not None else None
        )

        return replace(
            self, height=new_height, width=new_width, intrinsics=new_intrinsics
        )


@dataclass(frozen=True)
class RGBStream(ImageStream):
    """RGB image stream. By convention, the dtype is uint8."""

    channels: int = 3
    channel_order: RGBChannelOrderType = "HWC"


@dataclass(frozen=True)
class DepthStream(ImageStream):
    """RGB image stream. By convention, the dtype is (usually) float32."""

    # orthogonal or perspective depth measurement
    orthogonal: bool = True
    channels = None
    channel_order: ChannelOrderType = "HW"

    def __post_init__(self):
        if self.channels is not None:
            raise ValueError("channels must be None for a depth stream")
        if self.channel_order != "HW":
            raise ValueError("Depth stream must have channel order HW")


@dataclass(frozen=True)
class CameraSpec(Spec):
    streams: frozendict[str, ImageStream]
    time: int | None = None
    intrinsics: PinholeCameraIntrinsic | None = None
    extrinsics: torch.Tensor | None = None
    # the key (within the obs dict) with the dynamic pose information for the camera
    # e.g. a wrist camera would have extrinsics relative to the end effector pose
    dynamic_pose_obs_key: str | tuple[str, ...] | None = None
    baseline: float | None = None

    def __init__(
        self,
        streams: Mapping[str, ImageStream],
        time: int | None = None,
        intrinsics: PinholeCameraIntrinsic | None = None,
        extrinsics: torch.Tensor | None = None,
        dynamic_pose_obs_key: str | tuple[str, ...] | None = None,
        baseline: float | None = None,
    ):
        streams = {
            key: replace(
                stream,
                time=time,
                # if intrinsics not set for the stream, use the camera intrinsics
                intrinsics=stream.intrinsics or intrinsics,
            )
            for key, stream in streams.items()
        }

        object.__setattr__(self, "streams", frozendict(streams))
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "extrinsics", extrinsics)
        object.__setattr__(self, "dynamic_pose_obs_key", dynamic_pose_obs_key)
        object.__setattr__(self, "baseline", baseline)

    @property
    def shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    def replace(self, **kwargs) -> CameraSpec:
        """Replace the camera spec with a new one. This is used to change the
        time dimension of the camera spec."""

        new_streams = kwargs.pop("streams", None) or self.streams

        stream_kwargs = {}
        for key in ("time", "intrinsics"):
            if key in kwargs:
                stream_kwargs[key] = kwargs.pop(key)

        if stream_kwargs:
            new_streams = {
                key: replace(stream, **stream_kwargs)
                for key, stream in new_streams.items()
            }

        return replace(self, streams=new_streams, **kwargs)


@dataclass(frozen=True)
class PointCloudSpec(Spec):
    """The shape should be (*leading_dims) + (3 or 6,), depending on whether
    the point cloud has color or not."""

    feature_dim: int
    time: int | None = None
    color: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        raise ValueError("A pointcloud does not have a fixed shape.")


@dataclass(frozen=True)
class EmbedSpec:
    embed_dim: int
    n_tokens: int | None = None
    fixed_shape: bool = True

    def __post_init__(self):
        if self.fixed_shape and self.n_tokens is None:
            raise ValueError("Must specify n_tokens for an embedding of fixed shape")
        elif not self.fixed_shape and self.n_tokens is not None:
            raise ValueError("n_tokens must be None for an embedding of variable shape")

    @property
    def shape(self) -> tuple[int | None, int]:
        return (self.n_tokens, self.embed_dim)

    def concat(self, other: EmbedSpec) -> EmbedSpec:
        assert self.embed_dim == other.embed_dim
        fixed_shape = self.fixed_shape and other.fixed_shape
        n_tokens = self.n_tokens + other.n_tokens if fixed_shape else None  # type: ignore
        return EmbedSpec(
            embed_dim=self.embed_dim,
            n_tokens=n_tokens,
            fixed_shape=fixed_shape,
        )


@dataclass(frozen=True)
class ActionSpec(Spec):
    action_dim: int
    time: int | None = None
    a_mean: torch.Tensor | None = None
    a_var: torch.Tensor | None = None
    a_min: torch.Tensor | None = None
    a_max: torch.Tensor | None = None
    n_actions: int = 0

    def __init__(
        self,
        action_dim: int,
        time: int | None = None,
        a_mean: torch.Tensor | None = None,
        a_var: torch.Tensor | None = None,
        a_min: torch.Tensor | None = None,
        a_max: torch.Tensor | None = None,
        n_actions: int = 0,
        actions: torch.Tensor | None = None,
    ):
        object.__setattr__(self, "action_dim", action_dim)
        object.__setattr__(self, "time", time)

        mean = actions.mean(0) if actions is not None else a_mean
        var = actions.var(0) if actions is not None else a_var
        min = actions.min(0).values if actions is not None else a_min
        max = actions.max(0).values if actions is not None else a_max
        n_actions = actions.shape[0] if actions is not None else n_actions
        self._set_stats(mean, var, min, max, n_actions)

    @property
    def shape(self) -> tuple[int, ...]:
        if self.time is None:
            return (self.action_dim,)
        else:
            return (self.time, self.action_dim)

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

    def set_obs_seq_len(self, seq_len: int | None) -> DataSpecs:
        """Set the sequence length of the observation specs. This is used to
        create a new DataSpecs object with the same specs but with a different
        observation sequence length."""
        new_obs = {}
        for key, spec in self.obs.items():
            if isinstance(spec, CameraSpec):
                new_obs[key] = spec.replace(time=seq_len)
            elif isinstance(spec, ObsSpec):
                new_obs[key] = replace(spec, time=seq_len)
        return self.replace(obs=new_obs)

    def set_action_seq_len(self, seq_len: int | None) -> DataSpecs:
        """Set the sequence length of the action specs. This is used to create a
        new DataSpecs object with the same specs but with a different action
        sequence length."""
        return self.replace(action=replace(self.action, time=seq_len))

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
    def state_dim(self) -> int | None:
        try:
            robot_state_space = self.obs["robot_state"]
        except KeyError:
            return None

        return robot_state_space.shape[-1]

    @property
    def action_seq_len(self) -> int:
        assert len(self.action.shape) == 2
        return self.action.shape[0]

    @property
    def action_dim(self) -> int:
        assert len(self.action.shape) == 2
        return self.action.shape[-1]

    @property
    def obs_embed_dim(self) -> int | None:
        try:
            embed_space = self.obs["embed"]
        except KeyError:
            return None

        return embed_space.shape[-1]

    @property
    def obs_embed_seq_len(self) -> int | None:
        try:
            embed_space = self.obs["embed"]
        except KeyError:
            return None

        assert isinstance(embed_space, EmbedSpec)
        if embed_space.fixed_shape:
            return embed_space.n_tokens
        else:
            return None

    @property
    def goal_embed_seq_len(self) -> int | None:
        if self.goal is None:
            return None

        try:
            return self.goal["embed"].shape[0]
        except KeyError:
            return None

    @property
    def goal_embed_dim(self) -> int | None:
        if self.goal is None:
            return None

        try:
            return self.goal["embed"].shape[-1]
        except KeyError:
            return None


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


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # we don't want an explicit dependency on gymnasium in the specs module
    import gymnasium as gym


def specs_to_spaces(specs: DataSpecs) -> tuple["gym.Space", "gym.Space"]:
    """Convert specs to gym spaces."""
    import gymnasium.spaces as spaces

    obs_spaces = {key: spec_to_space(spec) for key, spec in specs.obs.items()}
    obs_space = spaces.Dict(obs_spaces)
    action_space = spec_to_space(specs.action)
    return obs_space, action_space


def spec_to_space(spec: Spec) -> "gym.Space":
    """Convert a single spec to a gym space."""
    import gymnasium.spaces as spaces

    if isinstance(spec, CameraSpec):
        subspaces = {
            name: spaces.Box(
                low=0,
                high=255,
                shape=stream.shape,
                dtype=np.float32 if isinstance(stream, DepthStream) else np.uint8,
            )
            for name, stream in spec.streams.items()
        }
        return spaces.Dict(subspaces)
    elif isinstance(spec, (ActionSpec, ObsSpec)):
        return spaces.Box(low=-np.inf, high=np.inf, shape=spec.shape, dtype=np.float32)
    else:
        raise ValueError(f"Unknown spec type: {type(spec).__name__}")
