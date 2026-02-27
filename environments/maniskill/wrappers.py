from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
from gymnasium.vector import VectorEnv, VectorObservationWrapper, VectorWrapper
from torch import Tensor

from environments.specs import (
    ActionSpec,
    CameraSpec,
    DataSpecs,
    DepthStream,
    EmbedSpec,
    PinholeCameraIntrinsic,
    RGBStream,
    TextSpec,
    space_to_spec,
)
from utils.math import convert_camera_frame_transform_convention

from .goals.instructions import ENV_INSTRUCTIONS

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv
    from mani_skill.envs.sapien_env import BaseEnv


log = logging.getLogger(__name__)

EMBEDDINGS_DIR = Path(__file__).parent / "goals" / "preprocessed_embeddings"


class ManiSkillVectorEnv(VectorEnv[dict[str, Tensor], Tensor, Tensor]):
    """
    Gymnasium Vector Env implementation for ManiSkill environments running on the GPU for parallel simulation and optionally parallel rendering

    Modified from mani_skill.vector.wrappers.gymnasium.ManiSkillVectorEnv to
    be compatible with gymnasium 1.0+, and to remove the auto_reset and
    record_metrics features that we handle in dedicated wrappers.

    Args:
        env: The environment created via gym.make / after wrappers are applied.
        ignore_terminations (bool): Whether this wrapper ignores terminations when deciding when to auto reset. Terminations can be caused by
            the task reaching a success or fail state as defined in a task's evaluation function. Default is False, meaning there is early stop in
            episode rollouts. If set to True, this would generally for situations where you may want to model a task as infinite horizon where a task
            stops only due to the timelimit.
    """

    def __init__(
        self,
        env: gym.Env[dict[str, Tensor], Tensor],
        ignore_terminations: bool = False,
    ):
        self._env = env
        num_envs = self.base_env.num_envs
        self.num_envs = num_envs
        self.ignore_terminations = ignore_terminations
        self.spec = self._env.spec

        self.single_observation_space = self._env.get_wrapper_attr(
            "single_observation_space"
        )
        self.single_action_space = self._env.get_wrapper_attr("single_action_space")
        self.action_space = self._env.get_wrapper_attr("action_space")
        self.observation_space = self._env.get_wrapper_attr("observation_space")
        self.metadata = self._env.metadata
        self.metadata.update(autoreset_mode=gym.vector.AutoresetMode.DISABLED)

    @property
    def device(self):
        return self.base_env.device

    @property
    def base_env(self) -> BaseEnv:
        return self._env.unwrapped

    @property
    def unwrapped(self):
        return self.base_env

    @property
    def render_mode(self) -> str:
        return self.base_env.render_mode

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ):
        return self._env.reset(seed=seed, options=options)

    def step(
        self, actions: Tensor
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Any]]:
        obs, rew, terminations, truncations, infos = self._env.step(actions)

        if isinstance(terminations, bool):
            terminations = torch.tensor([terminations], device=self.device)

        if self.ignore_terminations:
            terminations[:] = False

        return obs, rew, terminations, truncations, infos

    def close_extras(self, **kwargs: Any):
        self._env.close()

    def call(self, name: str, *args, **kwargs) -> tuple[Any, ...]:
        function = getattr(self._env, name)
        return function(*args, **kwargs)

    def get_attr(self, name: str):
        raise RuntimeError(
            "To get an attribute get it from the .env property of this object"
        )

    def render(self) -> np.ndarray:
        # Rendering is only used for human visualization or for video recording,
        # so we need to move it to the CPU at some point anyway.
        return self.base_env.render().cpu().numpy()


class ManiSkillGoalPosWrapper(VectorObservationWrapper):
    def observations(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        env_state = self.unwrapped.get_state_dict()
        if "goal_region" in env_state["actors"]:
            goal_region = env_state["actors"]["goal_region"]
            observations["extra"]["goal_pos"] = goal_region[..., :3]
        return observations


class ManiSkillGoalPosWrapper(VectorObservationWrapper):
    def observations(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        env_state = self.unwrapped.get_state_dict()
        if "goal_region" in env_state["actors"]:
            goal_region = env_state["actors"]["goal_region"]
            observations["extra"]["goal_pos"] = goal_region[..., :3]
        return observations


class ManiSkillSpecs(VectorWrapper):
    def __init__(
        self,
        env: VectorEnv[dict[str, Tensor], Tensor, Tensor],
        embeddings_dir: os.PathLike = EMBEDDINGS_DIR,
    ):
        super().__init__(env)

        self.env_id = self.env.unwrapped.spec.id
        self.goal_text = ENV_INSTRUCTIONS[self.env_id]

        embedding_path = Path(embeddings_dir) / f"{self.env_id}.pt"
        if not embedding_path.exists():
            raise FileNotFoundError(
                f"Could not find goal embedding for '{self.env_id}' at {embedding_path}"
            )

        log.info(f"Loading goal embedding from {embedding_path}")
        goal_embedding = torch.load(embedding_path, map_location=self.unwrapped.device)
        assert goal_embedding.ndim == 3
        self.goal_embedding = goal_embedding.expand(self.num_envs, -1, -1)

        action_spec = ActionSpec(
            action_dim=env.action_space.shape[-1], time=env.action_space.shape[-2]
        )

        wrapped_space = env.observation_space
        assert isinstance(wrapped_space, spaces.Dict)
        # we need to sample an observation to get the camera intrinsics
        reset_obs, _ = self.env.reset()

        observation_space = {
            "joint_pos": wrapped_space["agent"]["qpos"],
            "joint_vel": wrapped_space["agent"]["qvel"],
            "ee_pose": wrapped_space["extra"]["tcp_pose"],
        }

        if "goal_pos" in wrapped_space["extra"].keys():
            observation_space["goal_pos"] = wrapped_space["extra"]["goal_pos"]

        obs_specs = {
            key: space_to_spec(space) for key, space in observation_space.items()
        }

        camera_spaces = wrapped_space["sensor_data"]
        for cam_name, cam_space in camera_spaces.items():
            streams = {}

            n_envs, time, height, width, n_channels = cam_space["rgb"].shape
            assert n_envs == self.num_envs
            assert n_channels == 3
            streams["rgb"] = RGBStream(
                channels=3, channel_order="HWC", height=height, width=width
            )

            if "depth" in cam_space.keys():
                assert (
                    self.num_envs,
                    time,
                    height,
                    width,
                    1,
                ) == cam_space["depth"].shape
                streams["depth"] = DepthStream(height=height, width=width)

            assert height is not None and width is not None

            intrinsics = reset_obs["sensor_param"][cam_name]["intrinsic_cv"]
            assert (intrinsics[0] == intrinsics).all()
            intrinsics = intrinsics[0, 0]  # strip batch and singleton leading dims

            extrinsics = torch.eye(4, dtype=torch.float32, device=self.unwrapped.device)
            intrinsics = PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height, width
            )

            obs_specs[cam_name] = CameraSpec(
                streams=streams,
                time=time,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                dynamic_pose_obs_key=cam_name + "_transform",
            )

            extrinsics_space = wrapped_space["sensor_param"][cam_name]["cam2world_gl"]
            observation_space[cam_name + "_transform"] = extrinsics_space
            obs_specs[cam_name + "_transform"] = space_to_spec(extrinsics_space)

        goal_embed_spec = EmbedSpec(
            embed_dim=self.goal_embedding.shape[-1],
            n_tokens=self.goal_embedding.shape[-2],
        )
        goal_specs = {
            "env_id": TextSpec(),
            "description": TextSpec(),
            "embed": goal_embed_spec,
        }

        self.observation_space = spaces.Dict(observation_space)
        self.specs = DataSpecs(obs=obs_specs, action=action_spec, goal=goal_specs)

    def observations(self, observations: dict[str, Tensor]) -> dict[str, Tensor]:
        processed_obs = {
            "joint_pos": observations["agent"]["qpos"],
            "joint_vel": observations["agent"]["qvel"],
            "ee_pose": observations["extra"]["tcp_pose"],
        }

        if "goal_pos" in observations["extra"]:
            processed_obs["goal_pos"] = observations["extra"]["goal_pos"]

        for cam_name, cam_obs in observations["sensor_data"].items():
            cam_obs = dict(cam_obs)  # make a shallow copy

            if "depth" in cam_obs:
                # convert from int16 value in mm to float32 value in meters
                # remove singleton channel dimension
                cam_obs["depth"] = cam_obs["depth"].squeeze(-1) / 1000.0

            processed_obs[cam_name] = cam_obs

            extrinsics = observations["sensor_param"][cam_name]["cam2world_gl"]
            extrinsics = convert_camera_frame_transform_convention(
                extrinsics, origin="opengl", target="ros"
            )
            processed_obs[cam_name + "_transform"] = extrinsics

        return processed_obs

    def info(self, info: dict[str, Any]) -> dict[str, Any]:
        # remove this boolean value which does not have a compatible batch size
        info.pop("reconfigure", None)

        info["goal"] = {
            "env_id": [self.env_id for _ in range(self.num_envs)],
            "description": [self.goal_text for _ in range(self.num_envs)],
            "embed": self.goal_embedding,
        }

        return info

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
