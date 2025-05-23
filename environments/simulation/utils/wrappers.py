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
    ObsSpec
)

class IsaacLabPreProcess(gym.Wrapper):
    def __init__(self, env, obs_seq_len):
        super().__init__(env)

        # import it here, otherwise not possible to import before Isaac-Sim starts
        from isaaclab.utils.math import matrix_from_quat, make_pose   

        self._obs_seq_len = obs_seq_len

        # IK-Rel-controll uses 6d-delta pose + 1 gripper dim, but the model output 8d (3d-pos, 4d-quat, 1d-gripper)
        action_spec = ActionSpec(action_dim=self.env.unwrapped.single_action_space.shape[-1] + 1, time=1)

        self.env.reset()

        # prepare the observation structure to match the input structure all models expects.
        obs_spec = {}
        for obs_group_name in self.env.unwrapped.single_observation_space.keys():
            match obs_group_name:
                
                # pre-process only relevant robot state information
                case "proprioception":
                    joint_pos_dim = self.env.unwrapped.single_observation_space[obs_group_name]["joint_pos"].shape[-1]
                    gripper_pos_dim = self.env.unwrapped.single_observation_space[obs_group_name]["gripper_pos"].shape[-1]
                    obs_spec["robot_state"] = ObsSpec(elem_shape=(joint_pos_dim + gripper_pos_dim,), time=1)

                # extract all camera information
                case obs_group_name if "cam" in obs_group_name:
                    camera_cfg = {"streams": {},
                                  "extrinsics": None,
                                  "intrinsics": None}
                    
                    # extract all existing streams per camera (e.g. rgb, depth)
                    for obs_group_stream_name in self.env.unwrapped.single_observation_space[obs_group_name].keys():
                        match obs_group_stream_name:

                            case "rgb":
                                camera_cfg["streams"]["rgb"] = None
                                camera_cfg["streams"]["rgb"] = RGBStream(
                                    channels=self.env.unwrapped.single_observation_space[obs_group_name]["rgb"].shape[-1],
                                    channel_order="CHW", # per default IsaacLab returns HWC, but all RGB channels are automatically permuted to CHW. Check the self.preprocess_obs method.
                                    height=self.env.unwrapped.single_observation_space[obs_group_name]["rgb"].shape[0],
                                    width=self.env.unwrapped.single_observation_space[obs_group_name]["rgb"].shape[1],
                                )

                            case "depth":
                                camera_cfg["streams"]["depth"] = None
                                camera_cfg["streams"]["depth"] = DepthStream(
                                    orthogonal = True,
                                    channels=None, # should be set to None, check DepthStream since it is assumed that depth image do not have a channel dim
                                    channel_order="HW",
                                    height=self.env.unwrapped.single_observation_space[obs_group_name]["depth"].shape[0],
                                    width=self.env.unwrapped.single_observation_space[obs_group_name]["depth"].shape[1],
                                )

                    # extract camera extrinsics
                    pos_w = self.env.unwrapped.scene[obs_group_name].data.pos_w.clone()
                    rot_quat_w = self.env.unwrapped.scene[obs_group_name].data.quat_w_ros.clone()
                    rot_matrix_w = matrix_from_quat(rot_quat_w)
                    # current dataset preprocessing pipeline expects flattened homogenious matrices
                    extrinsic_matrix = make_pose(pos=pos_w, rot=rot_matrix_w).squeeze(0).cpu() # remove environment dim
                    camera_cfg["extrinsics"] = extrinsic_matrix

                    # extract camera intrinsics
                    camera_cfg["intrinsics"] = self.env.unwrapped.scene[obs_group_name]._data.intrinsic_matrices.squeeze(0).cpu() # remove environment dimensions as PinholeCameraIntrinsic.from_intrinsic_matrix expects 2d-tensor
 
                    # create the final CameraSpec
                    obs_spec[obs_group_name] = CameraSpec(
                        streams=camera_cfg["streams"],
                        time=self._obs_seq_len,
                        intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                            camera_cfg["intrinsics"],
                            height=self.env.unwrapped.single_observation_space[obs_group_name]["rgb"].shape[0], # !!!ASSUMPTION!!! All camera modalities have the same resolution
                            width=self.env.unwrapped.single_observation_space[obs_group_name]["rgb"].shape[1], # !!!ASSUMPTION!!! All camera modalities have the same resolution
                        ),
                        extrinsics=camera_cfg["extrinsics"],
                    )

        obs_spec["gripper_cam"] = obs_spec["gripper_cam"].replace(dynamic_pose_obs_key=("gripper_cam", "extrinsics"), extrinsics=torch.eye(4))

        self.specs = DataSpecs(
            obs=obs_spec,
            action=action_spec
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._preprocess_obs(obs)
        info = self._preprocess_info(info, device=self.env.unwrapped.device, batch_dim=self.env.unwrapped.scene.num_envs)
        return obs, info

    def step(self, action):
        action = self.pre_process_action_orientation(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._preprocess_obs(obs)
        info = self._preprocess_info(info, device=self.env.unwrapped.device, batch_dim=self.env.unwrapped.scene.num_envs)
        return obs, reward, terminated, truncated, info

    def _preprocess_obs(self, obs):
        pre_processed_obs = obs.copy()
        for obs_group_name, obs_group in obs.items():

            match obs_group_name:

                case "proprioception":
                    joint_pos = obs_group["joint_pos"]
                    gripper_pos = obs_group["gripper_pos"]
                    pre_processed_obs["robot_state"] = torch.cat((joint_pos, gripper_pos), dim=1)
                    
                
                case obs_group_name if "cam" in obs_group_name:
                    for obs_element_name, obs_element in obs_group.items():
                        match obs_element_name:
                            case "rgb":
                                # [B, H, W, C] -> [B, C, H, W]
                                pre_processed_obs[obs_group_name][obs_element_name] = obs_element.permute(0, 3, 1, 2)
                            case "depth":
                                # TODO: Remove: Balazs need to undo the permute anyway for depth images
                                # [B, H, W, C] -> [B, C, H, W]
                                pre_processed_obs[obs_group_name][obs_element_name] = obs_element.permute(0, 3, 1, 2)
                                pre_processed_obs[obs_group_name][obs_element_name] = pre_processed_obs[obs_group_name][obs_element_name].squeeze(0) # Current convetion: Depth images does not have a channel dim. Also, gym_env_dataset.step_return_to_tensor_dict() adds additional batch dims 


        return TensorDict(pre_processed_obs, device=self.env.unwrapped.device)

    def _preprocess_info(self, info, device, batch_dim):
        """IsaacLab returns different dictionaries on reset and step
        Make all elements vectorized tensors."""

        pre_processed_info = info["log"].copy()
        for k, v in pre_processed_info.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, device=device)
            if v.ndim == 0:
                v = v.view(batch_dim, -1)
            pre_processed_info[k] = v

        return pre_processed_info

    def pre_process_action_orientation(self, action: torch.Tensor):
        """Preprocess action orientation representation
        All models outputs action orientation as a quaternion, but IsaacLab environments expects euler angles"""

        # import it here, otherwise not possible to import before Isaac-Sim starts
        from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

        new_action = torch.zeros_like(action[:, :-1])
        # copy 3d-position
        new_action[:, :3] = action[:, :3] 
        # copy binary gripper action
        new_action[:, -1] = action[:, -1]
        # extract orientation as euler angles
        d_roll, d_pitch, d_yall = euler_xyz_from_quat(action[:, 3:7])
        d_roll = wrap_to_pi(d_roll)
        d_pitch = wrap_to_pi(d_pitch)
        d_yall = wrap_to_pi(d_yall)
        new_action[:, 3] = d_roll
        new_action[:, 4] = d_pitch
        new_action[:, 5] = d_yall

        return new_action
