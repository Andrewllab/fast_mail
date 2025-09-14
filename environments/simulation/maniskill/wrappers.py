import torch
import gymnasium as gym
from tensordict import TensorDict
from pathlib import Path
import logging

from environments.specs import (
    ActionSpec,
    CameraSpec,
    RGBStream,
    DepthStream,
    DataSpecs,
    PinholeCameraIntrinsic,
    ObsSpec,
    EmbedSpec,
)

log = logging.getLogger(__name__)


class ManiSkillPreProcess(gym.Wrapper):
    def __init__(self, env, obs_seq_len, embeddings_dir: str = "environments/simulation/maniskill/utils/preprocessed_embeddings"):
        super().__init__(env)
        self._obs_seq_len = obs_seq_len

        env_id = self.env.unwrapped.spec.id
        embedding_path = Path(embeddings_dir) / f"{env_id}.pt"

        if not embedding_path.exists():
            raise FileNotFoundError(f"Could not find goal embedding for '{env_id}' at {embedding_path}")
        
        log.info(f"Loading goal embedding from {embedding_path}")
        self.goal_embedding = torch.load(embedding_path)
        
        log.debug("Getting initial observation to read static camera intrinsics...")
        initial_obs, _ = self.env.reset()
        self._build_specs(initial_obs)

    def _build_specs(self, initial_obs: dict):
        log.debug("Building DataSpecs in __init__...")
        action_spec = ActionSpec(action_dim=self.env.action_space.shape[-1], time=1)
        
        
        obs_spec = {}

        goal_embedding_shape = self.goal_embedding.shape
        n_tokens = goal_embedding_shape[1]
        embed_dim = goal_embedding_shape[2]

        goal_embed_spec = EmbedSpec(embed_dim=embed_dim, n_tokens=n_tokens)
        goal_specs_dict = {"preprocessed_embed": goal_embed_spec}

        raw_obs_space = self.env.observation_space

        if "agent" in raw_obs_space.keys():
            qpos_shape = raw_obs_space["agent"]["qpos"].shape
            agent_dim = qpos_shape[-1]
            obs_spec["robot_state"] = ObsSpec(elem_shape=(agent_dim,), time=1)

        if "sensor_data" in raw_obs_space.keys():
            for cam_name, cam_space in raw_obs_space["sensor_data"].items():
                height, width = cam_space["rgb"].shape[1], cam_space["rgb"].shape[2]
                static_intrinsics_matrix = initial_obs["sensor_param"][cam_name][
                    "intrinsic_cv"
                ].squeeze(0)
                static_intrinsics = PinholeCameraIntrinsic.from_intrinsic_matrix(
                    static_intrinsics_matrix, height, width
                )

                extrinsics = initial_obs["sensor_param"][cam_name]["cam2world_gl"][0]

                streams = {}
                if "rgb" in cam_space.keys():
                    streams["rgb"] = RGBStream(
                        channels=3, channel_order="HWC", height=height, width=width
                    )
                if "depth" in cam_space.keys():
                    streams["depth"] = DepthStream(
                        orthogonal=False,
                        channels=None,
                        channel_order="HW",
                        height=height,
                        width=width,
                    )

                obs_spec[cam_name] = CameraSpec(
                    streams=streams,
                    time=self._obs_seq_len,
                    intrinsics=static_intrinsics,
                    extrinsics=extrinsics,
                    dynamic_pose_obs_key=("sensor_param", cam_name, "cam2world_gl"),
                )
        self.specs = DataSpecs(obs=obs_spec, action=action_spec, goal=goal_specs_dict)
        log.debug("Successfully built DataSpecs.")

    def _preprocess_obs(self, obs):
        processed_obs = {}

        batch_size = self.env.unwrapped.num_envs

        # TODO: can be handled better, but yeah I'm lazy :)
        processed_obs["goal"] = TensorDict(
            {"preprocessed_embed": self.goal_embedding.squeeze(0)}#, batch_size=[batch_size]
        )

        if "agent" in obs:
            robot_state = obs["agent"]["qpos"]
            if robot_state.ndim == 1:
                robot_state = robot_state.unsqueeze(0)
            processed_obs["robot_state"] = robot_state

        if "sensor_data" in obs:
            for cam_name, cam_obs in obs["sensor_data"].items():
                camera_td = {}
                if "rgb" in cam_obs:
                    camera_td["rgb"] = cam_obs["rgb"]
                if "depth" in cam_obs:
                    camera_td["depth"] = (
                        cam_obs["depth"].squeeze(-1) / 1000.0
                    )

                processed_obs[cam_name] = TensorDict(
                    camera_td, batch_size=[batch_size]
                )

        if "sensor_param" in obs:
            processed_obs["sensor_param"] = TensorDict.from_dict(
                obs["sensor_param"], batch_size=[batch_size]
            ).float()

        return TensorDict(processed_obs, batch_size=batch_size)

    def _preprocess_info(self, info: dict) -> TensorDict:
        processed_info = {k: v for k, v in info.items() if isinstance(v, torch.Tensor)}
        return TensorDict(
            processed_info, batch_size=self.env.unwrapped.num_envs
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._preprocess_obs(obs), self._preprocess_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward.shape[0] != self.env.unwrapped.num_envs:
            log.debug(f"Expanding reward shape from {reward.shape} to match num_envs.")
            reward = reward.expand(self.env.unwrapped.num_envs)
            terminated = terminated.expand(self.env.unwrapped.num_envs)
            truncated = truncated.expand(self.env.unwrapped.num_envs)

        processed_obs = self._preprocess_obs(obs)
        processed_info = self._preprocess_info(info)
        return processed_obs, reward, terminated, truncated, processed_info
