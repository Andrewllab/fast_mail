from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, SupportsFloat

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorWrapper
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)
from torch import Tensor

from environments.specs import (
    ActionSpec,
    CameraSpec,
    DataSpecs,
    DepthStream,
    PinholeCameraIntrinsic,
    RGBStream,
    TextSpec,
    space_to_spec,
)

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv

log = logging.getLogger(__name__)


class RoboCasaAdapter(
    gym.Wrapper[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], np.ndarray]
):

    # newer versions of gymnasium require render modes to be specified in metadata
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        env: gym.Env[dict[str, np.ndarray], np.ndarray],
        reduced_action_space: bool = False,
        render_cam_name: str | None = None,
        render_size: tuple[int, int] = (256, 256),
    ):
        super().__init__(env)

        if self.unwrapped.use_camera_obs:
            camera_names = self.unwrapped.camera_names

            self.all_camera_keys = [
                key
                for key in self.observation_space.spaces.keys()
                if any(key.startswith(cam) for cam in camera_names)
            ]

            obs_spaces = self.observation_space.spaces.copy()
            for cam_name in camera_names:
                # trim singleton channel dimension from depth images
                if f"{cam_name}_depth" in obs_spaces:
                    obs_spaces[f"{cam_name}_depth"] = spaces.Box(
                        low=-float("inf"),
                        high=float("inf"),
                        shape=obs_spaces[f"{cam_name}_depth"].shape[:-1],
                        dtype=np.float32,
                    )

                # add obs space for extrinsic transform for each camera
                extrinsics_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float64
                )
                obs_spaces[f"{cam_name}_transform"] = extrinsics_space

            self.observation_space = spaces.Dict(obs_spaces)

        if reduced_action_space:
            action_space = self.env.action_space
            low, high = action_space.low, action_space.high
            # In all robocasa environments, the action space is 12-dimensional
            # as the franka has a mobile platform.
            # We only control the first 7 dimensions, which correspond to the
            # end-effector and gripper.
            low, high = low[:7], high[:7]
            self.action_space = spaces.Box(low, high)
        self.reduced_action_space = reduced_action_space

        self.render_size = tuple(render_size)
        self.render_cam_name = render_cam_name or self.unwrapped.render_camera[0]

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        info["success"] = False
        return self.observation(obs), info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        action = self.action(action)
        observation, reward, terminated, truncated, info = self.env.step(action)

        success = bool(self.unwrapped._check_success())
        info["success"] = success
        terminated = terminated or success

        return self.observation(observation), reward, terminated, truncated, info

    def action(self, action: np.ndarray) -> np.ndarray:
        """Returns a modified action before :meth:`step` is called."""
        if self.reduced_action_space:
            # In all robocasa environments, the action space is 12-dimensional
            # as the franka has a mobile platform.
            # We only control the first 7 dimensions, which correspond to the
            # end-effector and gripper.
            expanded_action = np.zeros(12, dtype=np.float32)
            expanded_action[-1] = -1
            expanded_action[:7] = action
            return expanded_action
        else:
            return action

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Returns a modified observation."""
        if self.unwrapped.use_camera_obs:
            return self.cam_observation(observation)

        return observation

    @property
    def render_mode(self) -> str | None:
        if self.unwrapped.has_renderer:
            return "human"
        elif self.unwrapped.use_camera_obs or self.unwrapped.has_offscreen_renderer:
            return "rgb_array"
        else:
            return None

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            # TODO: cache rendered frames if both obs rendering and render
            # rendering share the same cameras?
            rendering = self.unwrapped.sim.render(
                camera_name=self.render_cam_name,
                height=self.render_size[0],
                width=self.render_size[1],
            )
            rendering = np.flip(rendering, axis=0)  # flip vertically
            return rendering
        else:
            return self.env.render()

    def cam_observation(
        self, observation: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        for key in self.all_camera_keys:
            value = observation[key]

            # flip rendered images vertically
            value = np.flip(value, axis=0)

            if key.endswith("_depth"):
                # remove singleton channel dimension
                value = value[..., 0]
                # convert from Mujoco simulation to depth in meters
                value = get_real_depth_map(self.unwrapped.sim, value)

            observation[key] = value

        for cam_name in self.unwrapped.camera_names:
            # add extrinsic transform for each camera to the obs dict
            extrinsics = get_camera_extrinsic_matrix(self.unwrapped.sim, cam_name)
            observation[f"{cam_name}_transform"] = extrinsics

        return observation

    def get_camera_intrinsic_matrix(self, camera_name: str) -> np.ndarray:
        camera_names = self.unwrapped.camera_names
        camera_idx = camera_names.index(camera_name)
        camera_height = self.unwrapped.camera_heights[camera_idx]
        camera_width = self.unwrapped.camera_widths[camera_idx]

        return get_camera_intrinsic_matrix(
            self.unwrapped.sim,
            camera_name,
            camera_height=camera_height,
            camera_width=camera_width,
        ).astype(np.float32)


class RoboCasaSpecs(VectorWrapper):
    def __init__(self, env: VectorEnv[dict[str, Tensor], Tensor, Tensor]):
        super().__init__(env)

        wrapped_space = env.observation_space
        assert isinstance(wrapped_space, spaces.Dict)

        camera_names = env.unwrapped.get_attr("camera_names")[0]

        # group rendered observations by camera based on their keys
        camera_streams = {
            cam_name: [
                key
                for key in wrapped_space.spaces.keys()
                if key.startswith(cam_name) and not key.endswith("_transform")
            ]
            for cam_name in camera_names
        }

        all_camera_keys = [key for keys in camera_streams.values() for key in keys]

        obs_specs = {
            key: space_to_spec(space)
            for key, space in wrapped_space.items()
            if key not in all_camera_keys
        }

        for cam_name in camera_names:
            streams = {}

            key = f"{cam_name}_image"
            n_envs, time, height, width, n_channels = wrapped_space[key].shape
            assert n_envs == self.num_envs
            assert n_channels == 3
            streams["rgb"] = RGBStream(
                channels=3, channel_order="HWC", height=height, width=width
            )

            if (key := f"{cam_name}_depth") in wrapped_space.keys():
                assert (
                    self.num_envs,
                    time,
                    height,
                    width,
                ) == wrapped_space[key].shape
                streams["depth"] = DepthStream(height=height, width=width)

            intrinsics = self.unwrapped.call(
                "get_camera_intrinsic_matrix", camera_name=cam_name
            )[0]

            # dynamic pose provides complete transform to camera
            extrinsics = torch.eye(4, dtype=torch.float32)
            intrinsics = PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height, width
            )

            obs_specs[cam_name] = CameraSpec(
                streams=streams,
                time=time,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                # this transform is added by the RoboCasaAdapter wrapper to the obs
                dynamic_pose_obs_key=f"{cam_name}_transform",
            )

        action_spec = ActionSpec(action_dim=self.single_action_space.shape[-1])

        goal_specs = {"description": TextSpec()}

        self.camera_streams = camera_streams
        self.specs = DataSpecs(obs=obs_specs, action=action_spec, goal=goal_specs)

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options or {})
        return self.observations(obs), self.info(info)

    def step(
        self, actions: Tensor
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return self.observations(obs), reward, terminated, truncated, self.info(info)

    def observations(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        for cam_name, stream_names in self.camera_streams.items():
            streams = {
                stream_name[len(cam_name) + 1 :]: observations.pop(stream_name)
                for stream_name in stream_names
            }

            # BackCompat: rename "image" stream to "rgb" for compatibility with
            # previous trained models, since the specs are frozen in the checkpoint
            if "image" in streams:
                streams["rgb"] = streams.pop("image")

            observations[cam_name] = streams

        return observations

    def info(self, info: dict[str, Any]) -> dict[str, Any]:
        ep_metas = self.unwrapped.call("get_ep_meta")
        descs = [ep_meta["lang"] for ep_meta in ep_metas]
        info["goal"] = {"description": descs}
        return info
