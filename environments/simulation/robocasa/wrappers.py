import logging
from typing import Any, Tuple
import os 
import imageio
import gymnasium as gym
import torch
from tensordict import TensorDict
import time

from environments.specs import (
    ActionSpec,
    CameraSpec,
    DataSpecs,
    DepthStream,
    PointMapStream,
    ObsSpec,
    PinholeCameraIntrinsic,
    RGBStream,
)
from utils.math import normalize

log = logging.getLogger(__name__)


import gym
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

        return obs_dict, reward, done, info

    def reset(self):
        obs_dict = self.env.reset()

        obs_dict = self.process_observation(obs_dict)

        return obs_dict

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

        action_spec = ActionSpec(
            action_dim=7, time=1
        )

        # prepare the observation structure to match the input structure all models expects.
        obs_spec = {}


        obs_spec["robot_state"] = ObsSpec(
            elem_shape=(9,), time=1
        )

        intrinsics = torch.zeros(3, 3)  # TODO: fill in correct intrinsics
        left_cam_pos = torch.tensor([[-0.5, 0.35, 1.05]])
        left_cam_rot_quat = torch.tensor([[0.55623853, 0.29935253, -0.37678665, -0.6775092]])  # w, x, y, z
        left_cam_rot_mat = matrix_from_quat(left_cam_rot_quat)
        extrinsics = make_pose(left_cam_pos, left_cam_rot_mat)
        left_cam = CameraSpec(
            streams={
                "rgb": RGBStream(224, 224, 3, channel_order="HWC"),
                "depth": DepthStream(224, 224, orthogonal=True),
                "pointmap": PointMapStream(224, 224, channels=3, channel_order="HWC"),
            },
            time=self._obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=224, width=224
            ),
            extrinsics=extrinsics,
        )


        right_cam_pos = torch.tensor([[-0.5, -0.35, 1.05]])
        right_cam_rot_quat = torch.tensor([[0.6775091886520386, 0.3767866790294647, -0.2993525564670563, -0.55623859167099]])  # w, x, y, z
        right_cam_rot_mat = matrix_from_quat(right_cam_rot_quat)
        extrinsics = make_pose(right_cam_pos, right_cam_rot_mat)
        # # correct extrinsics by adding conversion from ROS to WORLD camera convention
        # # we right-multiply, since we first need to transform the points
        # # into the WORLD convention, and then apply the extrinsics
        # extrinsics[:3, :3] = extrinsics[:3, :3] @ torch.tensor(
        #     ROS_TO_WORLD, dtype=extrinsics.dtype
        # )
        right_cam = CameraSpec(
            streams={
                "rgb": RGBStream(224, 224, 3, channel_order="HWC"),
                "depth": DepthStream(224, 224, orthogonal=True),
                "pointmap": PointMapStream(224, 224, channels=3, channel_order="HWC"),
            },
            time=self._obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=224, width=224
            ),
            extrinsics=extrinsics,
        )

        pos_gripper_cam = torch.tensor([[0.05, 0, 0]])
        rot_quat_gripper_cam = torch.tensor([[0, 0.707107, 0.707107, 0]])  # w, x, y, z
        rot_mat_gripper_cam = matrix_from_quat(rot_quat_gripper_cam)
        extrinsics = make_pose(pos_gripper_cam, rot_mat_gripper_cam)
        # # correct extrinsics by adding conversion from ROS to WORLD camera convention
        # # we right-multiply, since we first need to transform the points
        # # into the WORLD convention, and then apply the extrinsics
        # extrinsics[:3, :3] = extrinsics[:3, :3] @ torch.tensor(
        #     ROS_TO_WORLD, dtype=extrinsics.dtype
        # )
        gripper_cam = CameraSpec(
            streams={
                "rgb": RGBStream(224, 224, 3, channel_order="HWC"),
                "depth": DepthStream(224, 224, orthogonal=True),
                "pointmap": PointMapStream(224, 224, channels=3, channel_order="HWC"),
            },
            time=self._obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=224, width=224
            ),
            dynamic_pose_obs_key="gripper_cam_transform",
            # gripper_cam_transform provides complete transform to camera
            extrinsics=extrinsics,
        )

        obs_spec["left_cam"] = left_cam
        obs_spec["right_cam"] = right_cam
        obs_spec["gripper_cam"] = gripper_cam
        # set gripper_cam's extrinsics to the identity matrix at environment initialization, then at runtime read the dynamically-changing extrinsics for pointmap calculation.
        # For more details check: transforms/to_pointcloud.py
        obs_spec["gripper_cam"] = obs_spec["gripper_cam"].replace(
            dynamic_pose_obs_key=("gripper_cam", "extrinsics"), extrinsics=torch.eye(4)
        )

        self.specs = DataSpecs(obs=obs_spec, action=action_spec)

    def reset(self, **kwargs):
        # obs = self.env.reset(**kwargs)
        obs = self.env.reset()
        obs = self._preprocess_obs(obs)
        self._current_env_step = 0
        # info = self._preprocess_info(
        #     info,
        #     device=self.env.unwrapped.device,
        #     batch_dim=self.env.unwrapped.scene.num_envs,
        # )
        # if "eval_metrics" in obs:
        #     # eval metrics need to go into the info so they can be accumulated,
        #     # whereas all the obs except the last are dropped
        #     info["eval_metrics"] = obs.pop("eval_metrics")
        return obs, {}

    def step(self, action):
        action = self._preprocess_action(action)
        self._current_env_step += 1
        # try:
        obs, reward, terminated, info = self.env.step(action) 
        reward = torch.tensor([reward], dtype=torch.float32)
        terminated = torch.tensor([terminated], dtype=torch.bool,)
        truncated = torch.tensor([False], dtype=torch.bool,) # there is no truncated in robosuite envs

        if self._current_env_step >= self._max_steps_per_episode:
            truncated = torch.tensor([True], dtype=torch.bool,)
        
        # except Exception as e:
        #     log.warning(
        #         f"Exception during env.step(): {e}. Returning empty observation and failed trajectory."
        #     )
        #     obs = {}
        #     reward = torch.zeros(1, device=action.device)
        #     terminated = torch.ones(
        #         1, dtype=torch.bool, device=action.device
        #     )
        #     truncated = torch.zeros(
        #         1, dtype=torch.bool, device=action.device
        #     )
        #     info = {
        #         "Episode_Termination/IK_error": torch.ones(1)
        #     }
        #     info = TensorDict(info, device=action.device)
        #     info.auto_batch_size_(batch_dims=1)
        #     return obs, reward, terminated, truncated, info

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
                        "depth": torch.from_numpy(obs["robot0_agentview_left_depth"].squeeze(-1)).unsqueeze(0), # shape: (1, H, W, 1), float32
                        "pointmap": torch.from_numpy(obs["sampled_point_cloud"][:, :3]).unsqueeze(0),  # shape (1, H, W, 3), float32
                        "extrinsics": torch.from_numpy(get_camera_extrinsic_matrix(self.env.sim, "robot0_agentview_left")).unsqueeze(0),  # shape (1, 4, 4), float32
                        "intrinsics": torch.from_numpy(get_camera_intrinsic_matrix(self.env.sim, "robot0_agentview_left", 224, 224)).unsqueeze(0),  # shape (1, 3, 3), float32
                    },
                    
                    "right_cam": {
                        "rgb": torch.from_numpy(obs["robot0_agentview_right_image"].copy()).unsqueeze(0), #.permute(0, 3, 1, 2),  # shape: (1, H, W, 3), uint8
                        "depth": torch.from_numpy(obs["robot0_agentview_right_depth"].squeeze(-1)).unsqueeze(0), # shape: (1, H, W, 1), float32
                        "pointmap": torch.from_numpy(obs["sampled_point_cloud"][:, :3]).unsqueeze(0),   # shape (1, H, W, 3), float32
                        "extrinsics": torch.from_numpy(get_camera_extrinsic_matrix(self.env.sim, "robot0_agentview_right")).unsqueeze(0),  # shape (1, 4, 4), float32
                        "intrinsics": torch.from_numpy(get_camera_intrinsic_matrix(self.env.sim, "robot0_agentview_right", 224, 224)).unsqueeze(0),  # shape (1, 3, 3), float32
                    },

                    "gripper_cam": {
                        "rgb": torch.from_numpy(obs["robot0_eye_in_hand_image"].copy()).unsqueeze(0), #.permute(0, 3, 1, 2),   # shape: (1, H, W, 3), uint8
                        "depth": torch.from_numpy(obs["robot0_eye_in_hand_depth"].squeeze(-1)).unsqueeze(0), # shape: (1, H, W, 1), float32
                        "pointmap": torch.from_numpy(obs["sampled_point_cloud"][:, :3]).unsqueeze(0),  # shape (1, H, W, 3), float32
                        "extrinsics": torch.from_numpy(get_camera_extrinsic_matrix(self.env.sim, "robot0_eye_in_hand")).unsqueeze(0),  # shape (1, 4, 4), float32
                        "intrinsics": torch.from_numpy(get_camera_intrinsic_matrix(self.env.sim, "robot0_eye_in_hand", 224, 224)).unsqueeze(0),  # shape (1, 3, 3), float32
                    },
                    "ee_pose": ee_pose, # shape: (1, 7), float64
                    "robot_state": robot_state.float(), # shape: (1, 9), float64
                    "goal": {
                        "text": self.env.get_ep_meta()["lang"], # language description of the current task
                    }
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


class PointCloudWrapper(gym.Wrapper):
    def __init__(self, env, global_frame: bool, get_segmented_pc: bool = False, get_normal_pc: bool = True):
        super().__init__(env)
        # self.cam_names = [cam for cam in env.camera_names if cam != 'robot0_eye_in_hand']
        self.cam_names = env.camera_names
        self.pc_generator = PointCloudGenerator(env.sim, self.cam_names, env.camera_widths[0], env.camera_heights[0], global_frame)
        
        self.get_segmented_pc = get_segmented_pc
        self.get_normal_pc = get_normal_pc

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        
        if self.get_normal_pc:
            obs_dict["point_cloud"] = self.get_point_cloud(obs_dict)
        if self.get_segmented_pc:
            obs_dict["segmented_point_cloud"] = self.get_segmented_point_cloud(obs_dict)

        return obs_dict, reward, done, info
    
    def reset(self):
        obs_dict = self.env.reset()
        self.pc_generator.sim = self.env.sim
        
        if self.get_normal_pc:
            obs_dict["point_cloud"] = self.get_point_cloud(obs_dict)
        if self.get_segmented_pc:
            obs_dict["segmented_point_cloud"] = self.get_segmented_point_cloud(obs_dict)

        return obs_dict
    
    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)
        self.pc_generator.sim = self.env.sim
        
        if self.get_normal_pc:
            obs_dict["point_cloud"] = self.get_point_cloud(obs_dict)
        if self.get_segmented_pc:
            obs_dict["segmented_point_cloud"] = self.get_segmented_point_cloud(obs_dict)

        return obs_dict
    
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
    
    def _check_success(self):
        return self.env._check_success()


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
            ratio=0.0068,  # instead of K=num_points, corresponds to num_points=1024
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
        obs_dict, reward, done, info = self.env.step(action)

        obs_dict[self.key] = self.pc_sampler.sample(obs_dict["point_cloud"], self.num_points)
        return obs_dict, reward, done, info
    
    def reset(self):
        obs_dict = self.env.reset()
        
        obs_dict[self.key] = self.pc_sampler.sample(obs_dict["point_cloud"], self.num_points)
        return obs_dict
    
    def reset_to(self, state):
        obs_dict = self.env.reset_to(state)
        
        obs_dict[self.key] = self.pc_sampler.sample(obs_dict["point_cloud"], self.num_points)
        return obs_dict
    
    def _check_success(self):
        return self.env._check_success()


class VideoRecorder(gym.Wrapper):
    """
    Simple video recorder based on: https://github.com/robocasa/robocasa/blob/main/robocasa/utils/env_utils.py  
    It uses imageio to write mp4 files, one per episode.
    """

    def __init__(
        self,
        env: gym.Env,
        out_dir: str,
        filename_prefix: str,
        camera_name: str,
        size: Tuple,
        fps: int,
    ):
        super().__init__(env)
        self.out_dir = out_dir
        self.filename_prefix = filename_prefix
        self.camera_name = camera_name
        self.width, self.height = size
        self.fps = fps

        os.makedirs(self.out_dir, exist_ok=True)
        self._writer = None
        self._episode_idx = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # start new episode file
        if self._writer is not None:
            self._writer.close()
            self._writer = None  

        # open writer
        fname = f"{self.filename_prefix}_{self._episode_idx}.mp4"
        path = os.path.join(self.out_dir, fname)
        self._writer = imageio.get_writer(path, fps=self.fps)

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

        if terminated or truncated:
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            self._episode_idx += 1

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return super().close()
