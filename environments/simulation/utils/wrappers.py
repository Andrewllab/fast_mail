import torch
import gymnasium as gym
from tensordict import TensorDict

from environments.specs import (
    ActionSpec,
    CameraSpec,
    RGBStream,
    DepthStream,
    DataSpecs,
    PinholeCameraIntrinsic,
    ObsSpec,
)


class IsaacLabPreProcess(gym.Wrapper):
    def __init__(self, env, obs_seq_len):
        super().__init__(env)

        # import it here, otherwise not possible to import before Isaac-Sim starts
        from isaaclab.utils.math import matrix_from_quat, make_pose

        self._obs_seq_len = obs_seq_len

        action_spec = ActionSpec(
            action_dim=self.env.unwrapped.single_action_space.shape[-1], time=1
        )

        # prepare the observation structure to match the input structure all models expects.
        obs_spec = {}
        for obs_group_name in self.env.unwrapped.single_observation_space.keys():
            match obs_group_name:

                # pre-process only relevant robot state information
                case "proprioception":
                    joint_pos_dim = self.env.unwrapped.single_observation_space[
                        obs_group_name
                    ]["joint_pos"].shape[-1]
                    gripper_pos_dim = self.env.unwrapped.single_observation_space[
                        obs_group_name
                    ]["gripper_closure"].shape[-1]
                    obs_spec["robot_state"] = ObsSpec(
                        elem_shape=(joint_pos_dim + gripper_pos_dim,), time=1
                    )

                # extract all camera information
                case obs_group_name if "cam" in obs_group_name:
                    camera_cfg = {"streams": {}, "extrinsics": None, "intrinsics": None}

                    # extract all existing streams per camera (e.g. rgb, depth)
                    for (
                        obs_group_stream_name
                    ) in self.env.unwrapped.single_observation_space[
                        obs_group_name
                    ].keys():
                        match obs_group_stream_name:

                            case "rgb":
                                camera_cfg["streams"]["rgb"] = None
                                camera_cfg["streams"]["rgb"] = RGBStream(
                                    channels=self.env.unwrapped.single_observation_space[
                                        obs_group_name
                                    ][
                                        "rgb"
                                    ].shape[
                                        -1
                                    ],
                                    channel_order="CHW",  # Per default, IsaacLab returns RGB-images with dimensions [B, H, W, C]: https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html#rgb-and-rgba. However, the channel order is automatically permuted in the self.preprocess_obs at each environment step.
                                    height=self.env.unwrapped.single_observation_space[
                                        obs_group_name
                                    ]["rgb"].shape[0],
                                    width=self.env.unwrapped.single_observation_space[
                                        obs_group_name
                                    ]["rgb"].shape[1],
                                )

                            case "depth":
                                camera_cfg["streams"]["depth"] = None
                                camera_cfg["streams"]["depth"] = DepthStream(
                                    orthogonal=True,
                                    channels=None,  # should be set to None since a DepthStream assumes that a depth image does not have a channel dim
                                    channel_order="HW",  # Per default, IsaacLab returns RGB-images with dimensions [B, H, W, C]: https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html#depth-and-distances. However, the channel order is automatically permuted in the self.preprocess_obs at each environment step.
                                    height=self.env.unwrapped.single_observation_space[
                                        obs_group_name
                                    ]["depth"].shape[0],
                                    width=self.env.unwrapped.single_observation_space[
                                        obs_group_name
                                    ]["depth"].shape[1],
                                )

                    # extract camera extrinsics
                    pos_w = self.env.unwrapped.scene[obs_group_name].data.pos_w.clone()
                    rot_quat_w_ros = self.env.unwrapped.scene[
                        obs_group_name
                    ].data.quat_w_ros.clone()
                    rot_matrix = matrix_from_quat(rot_quat_w_ros)
                    # current dataset preprocessing-pipeline expects flattened homogenious matrices
                    extrinsic_matrix = (
                        make_pose(pos=pos_w, rot=rot_matrix).squeeze(0).cpu()
                    )  # remove environment dim
                    camera_cfg["extrinsics"] = extrinsic_matrix

                    # extract camera intrinsics
                    camera_cfg["intrinsics"] = (
                        self.env.unwrapped.scene[obs_group_name]
                        .data.intrinsic_matrices.squeeze(0)
                        .cpu()
                    )  # remove environment dimensions as PinholeCameraIntrinsic.from_intrinsic_matrix expects 2d-tensor

                    # create the final CameraSpec
                    obs_spec[obs_group_name] = CameraSpec(
                        streams=camera_cfg["streams"],
                        time=self._obs_seq_len,
                        intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                            camera_cfg["intrinsics"],
                            height=self.env.unwrapped.single_observation_space[
                                obs_group_name
                            ]["rgb"].shape[
                                0
                            ],  # !!!ASSUMPTION!!! All camera streams (rgb and depth) have the same resolution
                            width=self.env.unwrapped.single_observation_space[
                                obs_group_name
                            ]["rgb"].shape[
                                1
                            ],  # !!!ASSUMPTION!!! All camera streams (rgb and depth) have the same resolution
                        ),
                        extrinsics=camera_cfg["extrinsics"],
                    )

        # set gripper_cam's extrinsics to the identity matrix at environment initialization, then at runtime read the dynamically-changing extrinsics for pointmap calculation.
        # For more details check: transforms/to_pointcloud.py
        obs_spec["gripper_cam"] = obs_spec["gripper_cam"].replace(
            dynamic_pose_obs_key=("gripper_cam", "extrinsics"), extrinsics=torch.eye(4)
        )

        self.specs = DataSpecs(obs=obs_spec, action=action_spec)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._preprocess_obs(obs)
        info = self._preprocess_info(
            info,
            device=self.env.unwrapped.device,
            batch_dim=self.env.unwrapped.scene.num_envs,
        )
        return obs, info

    def step(self, action):
        action = self._preprocess_action_orientation(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._preprocess_obs(obs)
        info = self._preprocess_info(
            info,
            device=self.env.unwrapped.device,
            batch_dim=self.env.unwrapped.scene.num_envs,
        )
        return obs, reward, terminated, truncated, info

    def _preprocess_obs(self, obs):
        pre_processed_obs = obs.copy()
        for obs_group_name, obs_group in obs.items():

            match obs_group_name:

                case "proprioception":
                    joint_pos = obs_group["joint_pos"]
                    gripper_pos = obs_group["gripper_closure"]
                    pre_processed_obs["robot_state"] = torch.cat(
                        (joint_pos, gripper_pos), dim=1
                    )

                case obs_group_name if "cam" in obs_group_name:
                    for obs_element_name, obs_element in obs_group.items():
                        match obs_element_name:
                            case "rgb":
                                # Per default, IsaacLab returns RGB-images with dimensions [B, H, W, C]: https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html#rgb-and-rgba
                                # Permute [B, H, W, C] -> [B, C, H, W].
                                pre_processed_obs[obs_group_name][obs_element_name] = (
                                    obs_element.permute(0, 3, 1, 2)
                                )
                            case "depth":
                                # Per default, IsaacLab returns Depth-images with dimensions [B, H, W, C] https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html#depth-and-distances
                                # TODO: Remove: Balazs need to undo the permute anyway for depth images
                                # Permute [B, H, W, C] -> [B, C, H, W]
                                pre_processed_obs[obs_group_name][obs_element_name] = (
                                    obs_element.permute(0, 3, 1, 2)
                                )
                                pre_processed_obs[obs_group_name][
                                    obs_element_name
                                ] = pre_processed_obs[obs_group_name][
                                    obs_element_name
                                ].squeeze(
                                    0
                                )  # Current convetion: Depth images does not have a channel dim. Also, gym_env_dataset.step_return_to_tensor_dict() adds additional batch dims

        ee_pose = torch.cat(
            (
                pre_processed_obs["proprioception"]["eef_pos_w"],  # shape: (T, 3)
                pre_processed_obs["proprioception"]["eef_quat_w"],  # shape: (T, 4)
            ),
            dim=-1,
        )
        pre_processed_obs["ee_pose"] = ee_pose

        return TensorDict(
            pre_processed_obs,
            device=self.env.unwrapped.device,
            batch_size=pre_processed_obs[obs_group_name][obs_element_name].shape[0],
        )

    def _preprocess_info(self, info, device, batch_dim):
        """IsaacLab returns dictionaries with different dimensions on reset and step.
        Transform all elements vectorized tensors.
        """

        pre_processed_info = info["log"].copy()
        # !!!TODO: fix this in the environment config!!!
        pre_processed_info["Episode_Reward/success"] = pre_processed_info[
            "Episode_Reward/success"
        ].int()
        for k, v in pre_processed_info.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, device=device)
            if v.ndim == 0:
                v = v.view(batch_dim, -1)
            pre_processed_info[k] = v

        return TensorDict(pre_processed_info, batch_size=batch_dim)

    def _detect_ctrl_mode(self) -> str:
        """Extract the controller and its mode from the environment nam."""
        env_name = str(self.env.unwrapped.spec.id).lower()
        if "ik-rel" in env_name:
            return "ik_relative"
        if "ik-abs" in env_name:
            return "ik_absolute"
        raise ValueError(
            f"Not supported controll type. Currently, only IK-absolute and IK-relaltive controllers are supported in simulation."
        )

    def _preprocess_action_orientation(self, action: torch.Tensor):
        """Preprocesses the policy's raw output to match the simulation control's mode. This way policy's output gets decoupled from the simulation control mode.
        Args:
            action : torch.Tensor
                Raw policy output (B, 8): [x, y, z, qx, qy, qz, qw, grip]
                Position units: meters
                Orientation: quaternion (qx, qy, qz, qw)
                grip: scalar in [0/1] (binary or continuous as configured)
        Output:
            IK-relative (B, 7): [dx, dy, dz, droll, dpitch, dyaw, grip]
            IK-absolute (B, 8): [x, y, z, qx, qy, qz, qw, grip]
        """
        control_mode = self._detect_ctrl_mode()

        match control_mode:

            case "ik_relative":
                # import it here, otherwise not possible to import before Isaac-Sim starts
                from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

                pre_processed_action = torch.zeros_like(action[:, :-1])
                # copy 3d-position
                pre_processed_action[:, :3] = action[:, :3]
                # copy binary gripper action
                pre_processed_action[:, -1] = action[:, -1]
                # extract orientation as euler angles
                d_roll, d_pitch, d_yall = euler_xyz_from_quat(action[:, 3:7])
                # the computed euler angles are larger than pi for some rotations
                d_roll = wrap_to_pi(d_roll)
                d_pitch = wrap_to_pi(d_pitch)
                d_yall = wrap_to_pi(d_yall)
                pre_processed_action[:, 3] = d_roll
                pre_processed_action[:, 4] = d_pitch
                pre_processed_action[:, 5] = d_yall

            case "ik_absolute":
                pre_processed_action = action

            case _:
                raise ValueError(
                    f"Not supported controll type. Currently, only IK-absolute and IK-relaltive controllers are supported in simulation."
                )

        return pre_processed_action
