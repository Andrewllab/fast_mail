import logging
import os
from typing import Tuple

import gymnasium as gym
import imageio
import torch
from gymnasium import spaces
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


import abc

import numpy as np
import open3d as o3d
import robosuite.utils.transform_utils as T
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)
from robosuite.utils.transform_utils import (
    axisangle2quat,
    euler2mat,
    mat2quat,
    quat2axisangle,
    quat2mat,
)
from robosuite.wrappers import Wrapper
from torch_geometric.nn import fps

from utils.math import make_pose, matrix_from_quat


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
    def __init__(
        self,
        env,
        global_frame: bool,
        get_segmented_pc: bool = False,
        get_normal_pc: bool = True,
    ):
        super().__init__(env)
        # self.cam_names = [cam for cam in env.camera_names if cam != 'robot0_eye_in_hand']
        self.cam_names = env.camera_names
        self.pc_generator = PointCloudGenerator(
            env.sim,
            self.cam_names,
            env.camera_widths[0],
            env.camera_heights[0],
            global_frame,
        )

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
        segmentations = {
            cam: obs_dict[f"{cam}_segmentation_mask"] for cam in self.cam_names
        }

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
    def _sample(
        self, point_cloud: np.ndarray, num_points: int, device="cuda"
    ) -> np.ndarray:

        point_cloud = torch.from_numpy(point_cloud).to(device)
        num_points = torch.tensor([num_points]).to(device)
        # remember to only use coord to sample
        sampled_indices = fps(
            point_cloud[..., :3],
            # ratio=0.0068,  # instead of K=num_points, corresponds to num_points=1024
            ratio=0.007,  # instead of K=num_points, corresponds to num_points=1024
            random_start=True,
            batch_size=1,
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

        self.key = (
            "uniform_sampled_point_cloud"
            if isinstance(self.pc_sampler, UniformPointCloudSampler)
            else "sampled_point_cloud"
        )

    def step(self, action):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)

        obs_dict[self.key] = self.pc_sampler.sample(
            obs_dict["point_cloud"], self.num_points
        )
        return obs_dict, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs_dict, info = self.env.reset()

        obs_dict[self.key] = self.pc_sampler.sample(
            obs_dict["point_cloud"], self.num_points
        )
        return obs_dict, info

    def reset_to(self, state):
        obs_dict, info = self.env.reset_to(state)

        obs_dict[self.key] = self.pc_sampler.sample(
            obs_dict["point_cloud"], self.num_points
        )
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
                    columns=[
                        "epoch",
                        f"{self._num_recorded_episodes} episodes at {fps}fps",
                    ],
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
            frame = np.flip(
                self.env.sim.render(
                    height=self.height, width=self.width, camera_name=self.camera_name
                ),
                axis=0,
            )
            self._writer.append_data(frame)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._writer is not None:
            # robocasa envs return upside-down frames -> flip vertically
            frame = np.flip(
                self.env.sim.render(
                    height=self.height, width=self.width, camera_name=self.camera_name
                ),
                axis=0,
            )
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
