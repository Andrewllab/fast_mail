import logging
from typing import Tuple
import os 
import imageio
import gymnasium as gym
from gymnasium import spaces
import torch
from tensordict import TensorDict

from environments.specs import (
    ActionSpec,
    CameraSpec,
    DataSpecs,
    DepthStream,
    ObsSpec,
    PinholeCameraIntrinsic,
    RGBStream,
)
from utils.math import normalize

log = logging.getLogger(__name__)


import numpy as np
from robosuite.utils.transform_utils import (
    axisangle2quat,
    quat2axisangle,
    quat2mat,
    euler2mat,
    mat2quat,
)
from utils.math import make_pose, matrix_from_quat
from torch_geometric.nn import fps
import abc
import open3d as o3d
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix
import robosuite.utils.transform_utils as T
from robosuite.wrappers import Wrapper


# taken from and adapted: https://github.com/ARISE-Initiative/robosuite/blob/master/robosuite/wrappers/gym_wrapper.py
class GymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = "rgb_array"
    """
    Initializes a Gym wrapper for RoboCasa environments. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

    Args:
        env (RoboCasaEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.
        flatten_obs (bool):
            Whether to flatten the observation dictionary into a 1d array. Defaults to True.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, render_width, render_height, keys=None, flatten_obs=True):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # choose a default camera / size for video if not provided
        self._render_camera = (self.env.camera_names[0] if self.env.camera_names else "frontview")
        self._render_width = render_width
        self._render_height = render_height

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs = self.env.reset()

        # Whether to flatten the observation space
        self.flatten_obs: bool = flatten_obs

        if self.flatten_obs:
            flat_ob = self._flatten_obs(obs)
            self.obs_dim = flat_ob.size
            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)
        else:

            def get_box_space(sample):
                """Util fn to obtain the space of a single numpy sample data"""
                if np.issubdtype(sample.dtype, np.integer):
                    low = np.iinfo(sample.dtype).min
                    high = np.iinfo(sample.dtype).max
                elif np.issubdtype(sample.dtype, np.inexact):
                    low = float("-inf")
                    high = float("inf")
                else:
                    raise ValueError()
                return spaces.Box(low=low, high=high, shape=sample.shape, dtype=sample.dtype)

            self.observation_space = spaces.Dict({key: get_box_space(obs[key]) for key in self.keys})

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low, high)

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return "rgb_array"
    
    @property
    def unwrapped(self) -> str | None:
        return self.env

    def _process_observation(self, obs_dict):
        for key in obs_dict:
            if "image" in key or "segmentation" in key:
                obs_dict[key] = np.flip(obs_dict[key], axis=0)
            elif "depth" in key:
                obs_dict[key] = np.flip(obs_dict[key], axis=0)
                obs_dict[key] = self._depthimg2Meters(obs_dict[key])

        return obs_dict

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def _depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

    def _flatten_obs(self, obs_dict, verbose=False):
        """
        Filters keys of interest out and concatenate the information.

        Args:
            obs_dict (OrderedDict): ordered dictionary of observations
            verbose (bool): Whether to print out to console as observation keys are processed

        Returns:
            np.array: observations flattened into a 1d array
        """
        ob_lst = []
        for key in self.keys:
            if key in obs_dict:
                if verbose:
                    print("adding key: {}".format(key))
                ob_lst.append(np.array(obs_dict[key]).flatten())
        return np.concatenate(ob_lst)

    def _filter_obs(self, obs_dict) -> dict:
        """
        Filters keys of interest out of the observation dictionary, returning a filterd dictionary.
        """
        return {key: obs_dict[key] for key in self.keys if key in obs_dict}

    def render(self):
        """
        Return an rgb_array for RecordVideo. Uses offscreen renderer.
        """
        frame = self.env.sim.render(
            height=self._render_height,
            width=self._render_width,
            camera_name=self._render_camera,
        )
        # robocasa envs return upside-down frames -> flip vertically
        frame = np.flip(frame, axis=0)
        return frame
    
    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return observation instead of normal OrderedDict and optionally resets seed

        Returns:
            2-tuple:
                - (np.array) observations from the environment
                - (dict) an empty dictionary, as part of the standard return format
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()
        ob_dict = self._process_observation(ob_dict)
        # obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)
        return ob_dict, {}

    def reset_to(self, state):
        ob_dict = self.env.reset_to(state)

        ob_dict = self._process_observation(ob_dict)
        # obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)

        return ob_dict, {}

    def step(self, action):
        """
        Extends vanilla step() function call to return observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        ob_dict, reward, terminated, info = self.env.step(action)
        ob_dict = self._process_observation(ob_dict)

        # obs = self._flatten_obs(ob_dict) if self.flatten_obs else self._filter_obs(ob_dict)
        return ob_dict, reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()

    def close(self):
        """
        wrapper for calling underlying env close function
        """
        self.env.close()

    def _check_success(self):
        return self.env._check_success()


class RobosuiteWrapper(gym.Wrapper):
    """
    This wrapper does the following changes to the action and observation:
    - Flips the image and segmentation observations
    - Converts the depth observation from depth image to meters
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)

        obs_dict = self.process_observation(obs_dict)

        return obs_dict, reward, done, False, info

    def reset(self):
        obs_dict = self.env.reset()

        obs_dict = self.process_observation(obs_dict)

        return obs_dict, {}

    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)

        obs_dict = self.process_observation(obs_dict)

        return obs_dict

    def process_observation(self, obs_dict):
        for key in obs_dict:
            if "image" in key or "segmentation" in key:
                obs_dict[key] = np.flip(obs_dict[key], axis=0)
            elif "depth" in key:
                obs_dict[key] = np.flip(obs_dict[key], axis=0)
                obs_dict[key] = self.depthimg2Meters(obs_dict[key])

        return obs_dict

    def _check_success(self):
        return self.env._check_success()

    # https://github.com/htung0101/table_dome/blob/master/table_dome_calib/utils.py#L160
    def depthimg2Meters(self, depth):
        extent = self.sim.model.stat.extent
        near = self.sim.model.vis.map.znear * extent
        far = self.sim.model.vis.map.zfar * extent
        image = near / (1 - depth * (1 - near / far))
        return image

class RoboCasaPreProcess(gym.Wrapper):
    def __init__(self, env, obs_seq_len, max_steps_per_episode):
        super().__init__(env)

        self._obs_seq_len = obs_seq_len
        self._max_steps_per_episode = max_steps_per_episode
        self._current_env_step = 0

        obs_dict, _ = env.reset()

        joint_pos_dim = obs_dict["robot0_joint_pos"].shape[0]
        gripper_pos_dim = obs_dict["robot0_gripper_qpos"].shape[0]
        action_spec = ActionSpec(
            action_dim=joint_pos_dim + gripper_pos_dim, time=1
        )

        # prepare the observation structure to match the input structure all models expects.
        obs_spec = {}

        obs_spec["robot_state"] = ObsSpec(
            elem_shape=(joint_pos_dim + gripper_pos_dim,), time=1
        )

        left_cam_rgb_shape = obs_dict["robot0_agentview_left_image"].shape
        left_cam_depth_shape = obs_dict["robot0_agentview_left_depth"].shape
        left_cam_intrinsics = get_camera_intrinsic_matrix(self.unwrapped.sim, "robot0_agentview_left", camera_height=obs_dict["robot0_agentview_left_image"].shape[0], camera_width=obs_dict["robot0_agentview_left_image"].shape[1])
        left_cam_intrinsics = torch.from_numpy(left_cam_intrinsics).float() # otherwise RuntimeError: expected scalar type when torch.matmul in math/utils/transform_pointcloud
        # left_cam_extrinsics = get_camera_extrinsic_matrix(self.unwrapped.sim, "robot0_agentview_left")
        # left_cam_extrinsics = torch.from_numpy(left_cam_extrinsics).float() # otherwise RuntimeError: expected scalar type when torch.matmul in math/utils/transform_pointcloud
        left_cam_extrinsics = torch.eye(4, dtype=torch.float32)

        left_cam = CameraSpec(
            streams={
                "rgb": RGBStream(left_cam_rgb_shape[0], left_cam_rgb_shape[1], left_cam_rgb_shape[2], channel_order="HWC"),
                "depth": DepthStream(left_cam_depth_shape[0], left_cam_depth_shape[1], orthogonal=True),
            },
            time=self._obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                left_cam_intrinsics, height=left_cam_depth_shape[0], width=left_cam_depth_shape[1]
            ),
            dynamic_pose_obs_key="left_cam_transform",
            extrinsics=left_cam_extrinsics,
        )

        right_cam_rgb_shape = obs_dict["robot0_agentview_right_image"].shape
        right_cam_depth_shape = obs_dict["robot0_agentview_right_depth"].shape
        right_cam_intrinsics = get_camera_intrinsic_matrix(self.unwrapped.sim, "robot0_agentview_right", camera_height=obs_dict["robot0_agentview_right_image"].shape[0], camera_width=obs_dict["robot0_agentview_right_image"].shape[1])
        right_cam_intrinsics = torch.from_numpy(right_cam_intrinsics).float() # otherwise RuntimeError: expected scalar type when torch.matmul in math/utils/transform_pointcloud
        # right_cam_extrinsics = get_camera_extrinsic_matrix(self.unwrapped.sim, "robot0_agentview_right")
        # right_cam_extrinsics = torch.from_numpy(right_cam_extrinsics).float() # otherwise RuntimeError: expected scalar type when torch.matmul in math/utils/transform_pointcloud
        right_cam_extrinsics = torch.eye(4, dtype=torch.float32)

        right_cam = CameraSpec(
            streams={
                "rgb": RGBStream(right_cam_rgb_shape[0], right_cam_rgb_shape[1], right_cam_rgb_shape[2], channel_order="HWC"),
                "depth": DepthStream(right_cam_depth_shape[0], right_cam_depth_shape[1], orthogonal=True),
            },
            time=self._obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                right_cam_intrinsics, height=right_cam_depth_shape[0], width=right_cam_depth_shape[1]
            ),
            dynamic_pose_obs_key="right_cam_transform",
            extrinsics=right_cam_extrinsics,
        )

        gripper_cam_rgb_shape = obs_dict["robot0_eye_in_hand_image"].shape
        gripper_cam_depth_shape = obs_dict["robot0_eye_in_hand_depth"].shape
        gripper_cam_intrinsics = get_camera_intrinsic_matrix(self.unwrapped.sim, "robot0_eye_in_hand", camera_height=obs_dict["robot0_eye_in_hand_image"].shape[0], camera_width=obs_dict["robot0_eye_in_hand_image"].shape[1])
        gripper_cam_intrinsics = torch.from_numpy(gripper_cam_intrinsics).float() # otherwise RuntimeError: expected scalar type when torch.matmul in math/utils/transform_pointcloud
        # gripper_cam_extrinsics = get_camera_extrinsic_matrix(self.unwrapped.sim, "robot0_eye_in_hand")
        # gripper_cam_extrinsics = torch.from_numpy(gripper_cam_extrinsics).float() # otherwise RuntimeError: expected scalar type when torch.matmul in math/utils/transform_pointcloud
        gripper_cam_extrinsics = torch.eye(4, dtype=torch.float32)
        gripper_cam = CameraSpec(
            streams={
                "rgb": RGBStream(gripper_cam_rgb_shape[0], gripper_cam_rgb_shape[1], gripper_cam_rgb_shape[2], channel_order="HWC"),
                "depth": DepthStream(gripper_cam_depth_shape[0], gripper_cam_depth_shape[1], orthogonal=True),
            },
            time=self._obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                gripper_cam_intrinsics, height=gripper_cam_depth_shape[0], width=gripper_cam_depth_shape[1]
            ),
            dynamic_pose_obs_key="gripper_cam_transform",
            # gripper_cam_transform provides complete transform to camera
            extrinsics=gripper_cam_extrinsics,
        )

        obs_spec["left_cam"] = left_cam
        obs_spec["right_cam"] = right_cam
        obs_spec["gripper_cam"] = gripper_cam

        self.specs = DataSpecs(obs=obs_spec, action=action_spec)

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        obs = self._preprocess_obs(obs)
        self._current_env_step = 0

        return obs, info

    def step(self, action):
        action = self._preprocess_action(action)
        self._current_env_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action) 
        reward = torch.tensor([reward], dtype=torch.float32)
        
        assert not truncated  # raw value always False, there is no truncated in robosuite envs
        assert not terminated  # raw RoboCasa terminated values always False
        success = self._check_success()

        info["success"] = torch.tensor([success], dtype=torch.bool)
        # raw RoboCasa terminated values always False, so just use success as a termination.  
        terminated = torch.tensor([success], dtype=torch.bool)

        # necessary as robocasa/robosuite environment doesn't have a maximum episode steps
        if self._current_env_step >= self._max_steps_per_episode:
            truncated = torch.tensor([True], dtype=torch.bool)
        
        obs = self._preprocess_obs(obs)
        info = TensorDict(info)

        return obs, reward, terminated, truncated, info

    def _preprocess_obs(self, obs):

        # robosuite env does not return a env/batch dimension, so we add one everywhere!!!
        robot_state = torch.cat(
            (
                torch.from_numpy(obs["robot0_joint_pos"]).unsqueeze(0), # shape: (1, 7), robot joint positions
                torch.from_numpy(obs["robot0_gripper_qpos"]).unsqueeze(0) # shape (1, 2), gripper joint positions which corresponds to the degree of open/close of the gripper (same as gripper_closure in Isaac)
            ), 
            dim=-1
        )
        
        ee_pose = torch.cat(
            (
                torch.from_numpy(obs["robot0_eef_pos"]).unsqueeze(0),  # shape: (1, 3)
                torch.from_numpy(obs["robot0_eef_quat"]).unsqueeze(0),  # shape: (1, 4)
            ),
            dim=-1,
        )

        pre_processed_obs = TensorDict(
            {
                "left_cam": {
                    "rgb": torch.from_numpy(obs["robot0_agentview_left_image"].copy()).unsqueeze(0), #.permute(0, 3, 1, 2), # shape: (1, H, W, 3), uint8
                    "depth": torch.from_numpy(obs["robot0_agentview_left_depth"].squeeze(-1)).unsqueeze(0), # shape: (1, H, W), float32, later to_pontmap transform gets applied
                    # "pointcloud": torch.from_numpy(obs["sampled_point_cloud"][:, :3]).unsqueeze(0),  # shape (1, H, W, 3), float32
                    # "extrinsics": torch.from_numpy(get_camera_extrinsic_matrix(self.unwrapped.sim, "robot0_agentview_left")).unsqueeze(0),  # shape (1, 4, 4), float32
                    # "intrinsics": torch.from_numpy(get_camera_intrinsic_matrix(self.unwrapped.sim, "robot0_agentview_left", 224, 224)).unsqueeze(0),  # shape (1, 3, 3), float32
                },
                
                "right_cam": {
                    "rgb": torch.from_numpy(obs["robot0_agentview_right_image"].copy()).unsqueeze(0), #.permute(0, 3, 1, 2),  # shape: (1, H, W, 3), uint8
                    "depth": torch.from_numpy(obs["robot0_agentview_right_depth"].squeeze(-1)).unsqueeze(0), # shape: (1, H, W), float32, later to_pontmap transform gets applied
                    # "pointcloud": torch.from_numpy(obs["sampled_point_cloud"][:, :3]).unsqueeze(0),   # shape (1, H, W, 3), float32
                    # "extrinsics": torch.from_numpy(get_camera_extrinsic_matrix(self.unwrapped.sim, "robot0_agentview_right")).unsqueeze(0),  # shape (1, 4, 4), float32
                    # "intrinsics": torch.from_numpy(get_camera_intrinsic_matrix(self.unwrapped.sim, "robot0_agentview_right", 224, 224)).unsqueeze(0),  # shape (1, 3, 3), float32
                },

                "gripper_cam": {
                    "rgb": torch.from_numpy(obs["robot0_eye_in_hand_image"].copy()).unsqueeze(0), #.permute(0, 3, 1, 2),   # shape: (1, H, W, 3), uint8
                    "depth": torch.from_numpy(obs["robot0_eye_in_hand_depth"].squeeze(-1)).unsqueeze(0), # shape: (1, H, W), float32, later to_pontmap transform gets applied
                    # "pointcloud": torch.from_numpy(obs["sampled_point_cloud"][:, :3]).unsqueeze(0),  # shape (1, H, W, 3), float32
                    # "extrinsics": torch.from_numpy(get_camera_extrinsic_matrix(self.unwrapped.sim, "robot0_eye_in_hand")).unsqueeze(0),  # shape (1, 4, 4), float32
                    # "intrinsics": torch.from_numpy(get_camera_intrinsic_matrix(self.unwrapped.sim, "robot0_eye_in_hand", 224, 224)).unsqueeze(0),  # shape (1, 3, 3), float32
                },
                "ee_pose": ee_pose, # shape: (1, 7), float64
                "robot_state": robot_state.float(), # shape: (1, 9), float64
                "goal": {
                    "text": np.array([self.unwrapped.get_ep_meta()["lang"]]), # language description of the current task
                },
                "left_cam_transform": torch.from_numpy(get_camera_extrinsic_matrix(self.unwrapped.sim, "robot0_agentview_left")).unsqueeze(0),
                "right_cam_transform": torch.from_numpy(get_camera_extrinsic_matrix(self.unwrapped.sim, "robot0_agentview_right")).unsqueeze(0),
                "gripper_cam_transform": torch.from_numpy(get_camera_extrinsic_matrix(self.unwrapped.sim, "robot0_eye_in_hand")).unsqueeze(0),
            },  # type: ignore
            batch_size=1
        )

        pre_processed_obs.auto_batch_size_(batch_dims=1)

        return pre_processed_obs

    def _preprocess_action(self, action):

        # in all robocasa environments, the action space is 12-dimensional as the franka has a mobile platform
        new_action = torch.zeros(12, dtype=torch.float32)
        new_action[-1] = -1  # use only the franka arm and hand joint, fix everything else: https://github.com/ALRhub/X_IL/issues/2
        new_action[:7] = action

        # robosuite expects numpy arrays as actions
        return new_action.numpy()
    
    def _check_success(self):
        return self.unwrapped._check_success()

class PointCloudGenerator:
    def __init__(
        self,
        sim,
        cam_names: list[str],
        img_width: int,
        img_height: int,
        global_frame: bool,
    ):
        self.sim = sim
        self.cam_names = cam_names
        self.img_width = img_width
        self.img_height = img_height
        self.global_frame = global_frame

    def get_point_cloud(
        self, imgs: dict[str, np.ndarray], depths: dict[str, np.ndarray]
    ) -> np.ndarray:
        o3d_point_cloud = o3d.geometry.PointCloud()
        colors = []

        for cam in self.cam_names:
            colors.append(imgs[cam])

            cam_intrinsics = self._get_cam_intrinsic(
                cam, self.img_width, self.img_height
            )

            o3d_depth = o3d.geometry.Image(depths[cam])
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                o3d_depth, cam_intrinsics
            )

            cam_pose = get_camera_extrinsic_matrix(self.sim, cam)
            transformed_cloud = o3d_cloud.transform(cam_pose)

            if not self.global_frame:
                base_pos = self.sim.data.get_site_xpos(f"mobilebase0_center")
                base_rot = self.sim.data.get_site_xmat(f"mobilebase0_center")

                base_pose = T.pose_inv(T.make_pose(base_pos, base_rot))
                transformed_cloud = transformed_cloud.transform(base_pose)

            o3d_point_cloud += transformed_cloud

        pc_points = np.asarray(o3d_point_cloud.points)
        pc_colors = np.array(colors).reshape(-1, 3)

        pc = np.concatenate([pc_points, pc_colors], axis=1)
        return pc

    def get_segmented_point_cloud(
        self,
        imgs: dict[str, np.ndarray],
        depths: dict[str, np.ndarray],
        segmentation: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        o3d_clouds = {}
        colors = {}

        for cam in self.cam_names:
            for class_name in segmentation[cam]:
                if class_name not in o3d_clouds:
                    o3d_clouds[class_name] = []
                    colors[class_name] = []

                colors[class_name].append(imgs[cam][segmentation[cam][class_name]])

                depth = depths[cam].copy()
                depth[~segmentation[cam][class_name]] = (
                    -10
                )  # any invalid depth value is fine

                cam_intrinsics = self._get_cam_intrinsic(
                    cam, self.img_width, self.img_height
                )

                od_depth = o3d.geometry.Image(depth)
                o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                    od_depth, cam_intrinsics
                )

                cam_pose = get_camera_extrinsic_matrix(self.sim, cam)
                transformed_cloud = o3d_cloud.transform(cam_pose)

                if not self.global_frame:
                    base_pos = self.sim.data.get_site_xpos(f"mobilebase0_center")
                    base_rot = self.sim.data.get_site_xmat(f"mobilebase0_center")

                    base_pose = T.make_pose(base_pos, base_rot)
                    transformed_cloud = transformed_cloud.transform(
                        T.pose_inv(base_pose)
                    )

                o3d_clouds[class_name].append(transformed_cloud)

        class_to_point_cloud = {}
        for class_name, clouds in o3d_clouds.items():
            class_cloud = o3d.geometry.PointCloud()
            for cloud in clouds:
                class_cloud += cloud

            colors[class_name] = np.concatenate(colors[class_name])

            class_to_point_cloud[class_name] = np.concatenate(
                [np.asarray(class_cloud.points), colors[class_name]], axis=1
            )

        return class_to_point_cloud

    def _get_cam_intrinsic(self, cam_name: str, img_width: int, img_height: int):
        cam_mat = get_camera_intrinsic_matrix(
            sim=self.sim,
            camera_name=cam_name,
            camera_height=img_height,
            camera_width=img_width,
        )

        cx = cam_mat[0, 2]
        fx = cam_mat[0, 0]
        cy = cam_mat[1, 2]
        fy = cam_mat[1, 1]

        return o3d.camera.PinholeCameraIntrinsic(img_width, img_height, fx, fy, cx, cy)

    def _check_success(self):
        return self.env._check_success()

class PointCloudWrapper(gym.Wrapper):
    def __init__(self, env, global_frame: bool, get_segmented_pc: bool = False, get_normal_pc: bool = True):
        super().__init__(env)
        # self.cam_names = [cam for cam in env.camera_names if cam != 'robot0_eye_in_hand']
        self.cam_names = env.camera_names
        self.pc_generator = PointCloudGenerator(env.sim, self.cam_names, env.camera_widths[0], env.camera_heights[0], global_frame)
        
        self.get_segmented_pc = get_segmented_pc
        self.get_normal_pc = get_normal_pc

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        
        if self.get_normal_pc:
            obs_dict["point_cloud"] = self.get_point_cloud(obs_dict)
        if self.get_segmented_pc:
            obs_dict["segmented_point_cloud"] = self.get_segmented_point_cloud(obs_dict)

        return obs_dict, reward, terminated, truncated, info
    
    def reset(self):
        obs_dict, info = self.env.reset()
        self.pc_generator.sim = self.env.sim
        
        if self.get_normal_pc:
            obs_dict["point_cloud"] = self.get_point_cloud(obs_dict)
        if self.get_segmented_pc:
            obs_dict["segmented_point_cloud"] = self.get_segmented_point_cloud(obs_dict)

        return obs_dict, info
    
    def reset_to(self, state):
        obs_dict, info = self.env.reset_to(state)
        self.pc_generator.sim = self.env.sim
        
        if self.get_normal_pc:
            obs_dict["point_cloud"] = self.get_point_cloud(obs_dict)
        if self.get_segmented_pc:
            obs_dict["segmented_point_cloud"] = self.get_segmented_point_cloud(obs_dict)

        return obs_dict, info
    
    def get_point_cloud(self, obs_dict):
        assert self.get_normal_pc, "get_normal_pc is set to False"

        imgs = {cam: obs_dict[f"{cam}_image"] for cam in self.cam_names}
        depths = {cam: obs_dict[f"{cam}_depth"] for cam in self.cam_names}
        
        return self.pc_generator.get_point_cloud(imgs, depths)
    
    def get_segmented_point_cloud(self, obs_dict):
        assert self.get_segmented_pc, "get_segmented_pc is set to False"

        imgs = {cam: obs_dict[f"{cam}_image"] for cam in self.cam_names}
        depths = {cam: obs_dict[f"{cam}_depth"] for cam in self.cam_names}
        segmentations = {cam: obs_dict[f"{cam}_segmentation_mask"] for cam in self.cam_names}
        
        return self.pc_generator.get_segmented_point_cloud(imgs, depths, segmentations)


class BasePointCloudSampler(abc.ABC):
    def sample(self, point_cloud: np.ndarray, num_points: int) -> np.ndarray:
        if point_cloud.shape[0] <= num_points:
            return point_cloud

        return self._sample(point_cloud, num_points)

    @abc.abstractmethod
    def _sample(self, point_cloud: np.ndarray, num_points: int) -> np.ndarray:
        pass


class FPSPointCloudSampler(BasePointCloudSampler):
    def _sample(self, point_cloud: np.ndarray, num_points: int, device = "cuda") -> np.ndarray:

        point_cloud = torch.from_numpy(point_cloud).to(device)
        num_points = torch.tensor([num_points]).to(device)
        # remember to only use coord to sample
        sampled_indices = fps(
            point_cloud[..., :3], 
            # ratio=0.0068,  # instead of K=num_points, corresponds to num_points=1024
            ratio=0.007,  # instead of K=num_points, corresponds to num_points=1024
            random_start=True,
            batch_size=1 
            # batch=torch.zeros(point_cloud.shape[0], dtype=torch.long, device=device
        )
        point_cloud = point_cloud.squeeze(0).cpu().numpy()

        return point_cloud[sampled_indices.squeeze(0).cpu().numpy()]


class UniformPointCloudSampler(BasePointCloudSampler):
    def _sample(self, point_cloud: np.ndarray, num_points: int) -> np.ndarray:
        sampled_indices = np.random.choice(
            point_cloud.shape[0], num_points, replace=False
        )
        return point_cloud[sampled_indices]
    

class PointCloudSamplingWrapper(gym.Wrapper):
    def __init__(self, env, pc_sampler: BasePointCloudSampler, num_points: int):
        super().__init__(env)
        self.pc_sampler = pc_sampler
        self.num_points = num_points

        self.key = "uniform_sampled_point_cloud" if isinstance(self.pc_sampler, UniformPointCloudSampler) else "sampled_point_cloud"

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)

        obs_dict[self.key] = self.pc_sampler.sample(obs_dict["point_cloud"], self.num_points)
        return obs_dict, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs_dict, info = self.env.reset()
        
        obs_dict[self.key] = self.pc_sampler.sample(obs_dict["point_cloud"], self.num_points)
        return obs_dict, info
    
    def reset_to(self, state):
        obs_dict, info = self.env.reset_to(state)
        
        obs_dict[self.key] = self.pc_sampler.sample(obs_dict["point_cloud"], self.num_points)
        return obs_dict, info


class VideoRecorder(gym.Wrapper):
    """
    Simple video recorder based on: https://github.com/robocasa/robocasa/blob/main/robocasa/utils/env_utils.py  
    It uses imageio to write mp4 files, one per episode.
    """

    def __init__(
        self,
        env: gym.Env,
        num_eval_episodes: int,
        video_folder: str,
        filename_prefix: str,
        camera_name: str,
        size: Tuple,
        fps: int,
    ):
        super().__init__(env)
        self.video_folder = video_folder
        self.filename_prefix = filename_prefix
        self.camera_name = camera_name
        self.width, self.height = size
        self.fps = fps

        os.makedirs(self.video_folder, exist_ok=True)
        self._writer = None
        self._num_eval_episodes = num_eval_episodes
        self._num_recorded_episodes = 0

        try:
            import wandb

            if wandb.run is not None:
                self.rollouts_table = wandb.Table(
                    columns=["epoch", f"{self._num_recorded_episodes} episodes at {fps}fps"],
                    # log_mode="INCREMENTAL",
                )

        except ImportError:
            pass


    @property
    def run_name(self) -> str:
        if self._run_name is None:
            log.warning("run_name is not set for the video recorder")
            return "unknown_run"
        return self._run_name

    @run_name.setter
    def run_name(self, value: str):
        self._run_name = value

    @property
    def ckpt_epoch(self) -> int:
        if self._ckpt_epoch is None:
            log.warning("ckpt_epoch is not set for the video recorder")
            return 0
        return self._ckpt_epoch

    @ckpt_epoch.setter
    def ckpt_epoch(self, value: int):
        self._ckpt_epoch = value

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self._num_recorded_episodes >= self._num_eval_episodes:
            self.stop_recording()
            if self._num_eval_episodes >= self._num_recorded_episodes:
                self.upload_eval_runs_to_wandb()
        else:
            if self._writer is None:
                self.start_recording()
            # Don't stop recording as we want one mp4 file with all eval episodes per checkpoint
            self._num_recorded_episodes += 1

            # robocasa envs return upside-down frames -> flip vertically
            frame = np.flip(self.env.sim.render(height=self.height, width=self.width, camera_name=self.camera_name), axis=0)
            self._writer.append_data(frame)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._writer is not None:
            # robocasa envs return upside-down frames -> flip vertically
            frame = np.flip(self.env.sim.render(height=self.height, width=self.width, camera_name=self.camera_name), axis=0)
            self._writer.append_data(frame)            

        return obs, reward, terminated, truncated, info
    
    def start_recording(self):
        # open writer
        # fname = f"{self.filename_prefix}_{self._num_recorded_episodes}-{self._num_eval_episodes}.mp4"
        fname = f"{self.filename_prefix}.mp4"
        self.file_path = os.path.join(self.video_folder, fname)
        self._writer = imageio.get_writer(self.file_path, fps=self.fps)

    def stop_recording(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def upload_eval_runs_to_wandb(self):
        if self.rollouts_table is not None:
            import wandb

            video = wandb.Video(self.file_path, format="mp4")
            ckpt_epoch = self.ckpt_epoch
            self.rollouts_table.add_data(ckpt_epoch, video)

            assert wandb.run is not None
            wandb.run.log({"test/rollouts": self.rollouts_table})
