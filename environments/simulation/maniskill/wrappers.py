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

from utils.math import (
    convert_camera_frame_orientation_convention,
    make_pose,
    matrix_to_quaternion,
    quaternion_to_matrix,
    unmake_pose,
)

log = logging.getLogger(__name__)

def _convert_extrinsics_convention(
    extrinsics_gl: torch.Tensor, target: str = "world"
) -> torch.Tensor:
    """
    Converts a batch of 4x4 extrinsic from opengl to a target (ros or world).
    """
    pos, rot_mat_gl = unmake_pose(extrinsics_gl)

    quat_gl_wxyz = matrix_to_quaternion(rot_mat_gl)

    quat_target_wxyz = convert_camera_frame_orientation_convention(
        quat_gl_wxyz, origin="opengl", target=target
    )

    rot_mat_target = quaternion_to_matrix(quat_target_wxyz)

    extrinsics_target = make_pose(pos, rot_mat_target)

    return extrinsics_target

class ManiSkillPreProcess(gym.Wrapper):
    def __init__(self, env, obs_seq_len, embeddings_dir: str = "environments/simulation/maniskill/utils/preprocessed_embeddings"):
        super().__init__(env)
        self._obs_seq_len = obs_seq_len

        env_id = self.env.unwrapped.spec.id
        embedding_path = Path(embeddings_dir) / f"{env_id}.pt"

        if not embedding_path.exists():
            raise FileNotFoundError(f"Could not find goal embedding for '{env_id}' at {embedding_path}")
        
        log.info(f"Loading goal embedding from {embedding_path}")
        self.goal_embedding = torch.load(embedding_path, map_location=self.device)
        
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
        goal_specs_dict = {"embed": goal_embed_spec}

        raw_obs_space = self.env.observation_space

        if "agent" in raw_obs_space.keys():
            qpos_shape = raw_obs_space["agent"]["qpos"].shape
            agent_dim = qpos_shape[-1]
            obs_spec["robot_state"] = ObsSpec(elem_shape=(agent_dim,), time=1)

        if "sensor_data" in raw_obs_space.keys():
            for cam_name, cam_space in raw_obs_space["sensor_data"].items():
                if self.env.unwrapped.spec.id == "StackCube-v1" and cam_name == "hand_camera":
                    continue # Exclude hand cam from StackCube-v1
                height, width = cam_space["rgb"].shape[1], cam_space["rgb"].shape[2]
                static_intrinsics_matrix = initial_obs["sensor_param"][cam_name][
                    "intrinsic_cv"
                ].squeeze(0)
                static_intrinsics = PinholeCameraIntrinsic.from_intrinsic_matrix(
                    static_intrinsics_matrix, height, width
                )

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

                if cam_name == "hand_camera":
                    obs_spec[cam_name] = CameraSpec(
                        streams=streams,
                        time=self._obs_seq_len,
                        intrinsics=static_intrinsics,
                        extrinsics=torch.eye(4, dtype=torch.float32),
                        dynamic_pose_obs_key="gripper_cam_transform",
                    )
                else: 
                    extrinsics_gl = initial_obs["sensor_param"][cam_name]["cam2world_gl"][0]

                    extrinsics_ros = _convert_extrinsics_convention(
                        extrinsics_gl.unsqueeze(0), target="ros"
                    ).squeeze(0)

                    obs_spec[cam_name] = CameraSpec(
                        streams=streams,
                        time=self._obs_seq_len,
                        intrinsics=static_intrinsics,
                        extrinsics=extrinsics_ros,
                    )

        obs_spec["ee_pose"] = ObsSpec(elem_shape=(7,), time=1)
        obs_spec["gripper_cam_transform"] = ObsSpec(elem_shape=(4, 4), time=1)

        self.specs = DataSpecs(obs=obs_spec, action=action_spec, goal=goal_specs_dict)
        log.debug("Successfully built DataSpecs.")

    def _preprocess_obs(self, obs):
        processed_obs = {}

        batch_size = self.env.unwrapped.num_envs

        processed_obs["goal"] = TensorDict(
            {"embed": self.goal_embedding.squeeze(0)}
        )

        if "agent" in obs:
            robot_state = obs["agent"]["qpos"]
            if robot_state.ndim == 1:
                robot_state = robot_state.unsqueeze(0)
            processed_obs["robot_state"] = robot_state

        if "extra" in obs and "tcp_pose" in obs["extra"]:
            processed_obs["ee_pose"] = obs["extra"]["tcp_pose"]

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

            if "hand_camera" in obs["sensor_data"]:
                extrinsics_gl = obs["sensor_param"]["hand_camera"]["cam2world_gl"]
                extrinsics_ros = _convert_extrinsics_convention(extrinsics_gl, target="ros")
                processed_obs["gripper_cam_transform"] = extrinsics_ros

        # if "sensor_param" in obs:
        #     processed_obs["sensor_param"] = TensorDict.from_dict(
        #         obs["sensor_param"], batch_size=[batch_size]
        #     ).float()

        return TensorDict(processed_obs, batch_size=batch_size).to(self.device)

    def _preprocess_info(self, info: dict) -> TensorDict:
        processed_info = {k: v for k, v in info.items() if isinstance(v, torch.Tensor) and v.numel() == 1} # TODO: we can only log single values atm!
        return TensorDict(
            processed_info, batch_size=self.env.unwrapped.num_envs
        ).to(self.device)

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
        
        return (
            processed_obs,
            reward.to(self.device),
            terminated.to(self.device),
            truncated.to(self.device),
            processed_info,
        )