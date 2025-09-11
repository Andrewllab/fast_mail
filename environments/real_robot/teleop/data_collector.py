import gc
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import cv2
import numpy as np
import pygame
import torch
from scipy.spatial.transform import Rotation
from tensordict import TensorDict

from environments.real_robot.hardware.base_camera import BaseCamera
from environments.real_robot.hardware.utils.keyboard import KeyManager
from environments.real_robot.teleop.teleoperation_base import (
    RobotState,
    TeleoperationPair,
)

ExtrinsicsMatrix = List[List[float]]

log = logging.getLogger(__name__)


class CollectionData:
    joint_pos_list: List[torch.Tensor]
    joint_vel_list: List[torch.Tensor]
    ee_pos_list: List[torch.Tensor]
    ee_quat_list: List[torch.Tensor]
    ee_vel_list: List[torch.Tensor]
    gripper_state_list: List[torch.Tensor]
    dynamic_extrinsics_list: List[torch.Tensor]

    def __init__(self):
        self.joint_pos_list = []
        self.joint_vel_list = []
        self.ee_pos_list = []
        self.ee_quat_list = []
        self.ee_vel_list = []
        self.gripper_state_list = []
        self.dynamic_extrinsics_list = []

    def append_state(
        self, robot_state: RobotState, dynamic_extrinsics: torch.Tensor | None = None
    ):
        self.joint_pos_list.append(robot_state.joint_pos)
        self.joint_vel_list.append(robot_state.joint_vel)
        self.ee_pos_list.append(robot_state.ee_pos[:3])
        self.ee_quat_list.append(robot_state.ee_pos[3:])
        self.ee_vel_list.append(robot_state.ee_vel)
        self.gripper_state_list.append(robot_state.gripper_state)
        if dynamic_extrinsics is not None:
            self.dynamic_extrinsics_list.append(dynamic_extrinsics)


class DataCollectionManager:
    def __init__(
        self,
        teleoperation_pair: TeleoperationPair,
        data_dir: str,
        step_hz: int,
        cameras: Mapping[str, BaseCamera] | None = None,
        keep_stream_names: str | Sequence[str] | None = None,
        drop_stream_names: str | Sequence[str] | None = None,
        max_height_width: tuple[int, int] | None = None,
        baselines: Optional[dict[str, float]] = None,
        extrinsics: Optional[dict[str, ExtrinsicsMatrix]] = None,
    ):
        self.teleoperation_pair = teleoperation_pair
        self.cameras = cameras or {}

        if keep_stream_names is None:
            self.keep_stream_names = []
        elif isinstance(keep_stream_names, str):
            self.keep_stream_names = [keep_stream_names]
        else:
            self.keep_stream_names = list(keep_stream_names)

        if drop_stream_names is None:
            self.drop_stream_names = []
        elif isinstance(drop_stream_names, str):
            self.drop_stream_names = [drop_stream_names]
        else:
            self.drop_stream_names = list(drop_stream_names)

        self.baselines = baselines or {}
        self.extrinsics = (
            {
                name: np.array(extrinsics, dtype=np.float32)
                for name, extrinsics in (extrinsics or {}).items()
            }
            if extrinsics is not None
            else None
        )

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.clock = pygame.time.Clock()
        self.step_hz = step_hz
        self.episode_start = 0

        self.trajectory_robot_data: dict[str, CollectionData] = {}
        self.trajectory_cam_data: dict[str, dict[str, list]] = {}

        self.cam_calib: dict[str, dict] = {}
        self.resize_wh: dict[str, tuple[int, int] | None] = {
            cam_name: None for cam_name in self.cameras.keys()
        }
        self.crop_slices: dict[str, tuple[slice, slice] | None] = {
            cam_name: None for cam_name in self.cameras.keys()
        }
        for cam_name, cam in self.cameras.items():
            intr_dict = cam.get_intrinsics()
            fx = intr_dict["fx"]
            fy = intr_dict["fy"]
            cx = intr_dict["cx"]
            cy = intr_dict["cy"]
            height = intr_dict["height"]
            width = intr_dict["width"]

            if max_height_width is not None:
                max_height, max_width = max_height_width
                factor = max(max_height / height, max_width / width)
                factor = min(factor, 1.0)  # don't scale up, only down
                if factor < 1.0:
                    new_height = round(height * factor)
                    new_width = round(width * factor)
                    # opencv.resize expects (width, height)
                    self.resize_wh[cam_name] = (new_width, new_height)
                    fx *= factor
                    fy *= factor
                    cx *= factor
                    cy *= factor
                    log.info(
                        f"Resizing images from {cam_name} from {height}x{width} to {new_height}x{new_width}"
                    )
                    height, width = new_height, new_width

                if width > max_width or height > max_height:
                    if width > max_width:
                        left = round((width - max_width) / 2)
                        width_slice = slice(left, left + max_width)
                        cx -= left
                        width = max_width
                    else:
                        width_slice = slice(None)

                    if height > max_height:
                        top = round((height - max_height) / 2)
                        height_slice = slice(top, top + max_width)
                        cy -= top
                        height = max_height
                    else:
                        height_slice = slice(None)

                    self.crop_slices[cam_name] = (height_slice, width_slice)

                    log.info(
                        f"Center cropping images from {cam_name} to {height}x{width}"
                    )

            intrinsics = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
            )
            height_width = np.array([height, width], dtype=np.int32)
            self.cam_calib[cam_name] = {
                "intrinsics": intrinsics,
                "intrinsics_heights_width": height_width,
            }

            baseline = self.baselines.get(cam_name, intr_dict.get("baseline", None))
            if baseline is not None:
                self.cam_calib[cam_name]["baseline"] = baseline

            if self.extrinsics:
                self.cam_calib[cam_name]["extrinsics"] = self.extrinsics[cam_name]

    def start_key_listener(self):
        km = KeyManager()
        print("Press 'n' to collect new data or 'q' to quit data collection")
        counter = 0
        while km.key != "q":
            if km.key == "n":
                print()
                print("Preparing for new data collection")
                self.__create_empty_data()
                self.__reset_robots()
                time.sleep(3)  # Give some time for the robots to reset

                print("Start! Press 's' to save collected data or 'd' to discard.")

                while km.key not in ["s", "d"]:

                    self.__collection_step()
                    self.clock.tick(self.step_hz)
                    km.pool()

                else:
                    if km.key == "s":
                        print()
                        print("Saving data")

                        self.__save_data(counter)
                        self.__create_empty_data()
                        counter += 1
                        print("COUNTER", counter)

                        print("Saved!")
                    elif km.key == "d":
                        print()
                        print("Discarding collected data")
                        self.__create_empty_data()
                        print("Discarded!")

                    print(
                        "Press 'n' to collect new data or 'q' to quit data collection"
                    )

            km.pool()

        print()
        print("Ending data collection...")
        km.close()
        self.__close_hardware_connections()

    def __create_empty_data(self):
        """Allocate per-robot trajectory buffers and per-camera frame buffers."""
        self.trajectory_robot_data = {
            "leader": CollectionData(),
            "follower": CollectionData(),
        }

        self.trajectory_cam_data = {
            cam_name: defaultdict(list) for cam_name in self.cameras.keys()
        }

        self.episode_start = time.perf_counter()  # reset to current time

    def __reset_robots(self):
        self.teleoperation_pair.reset()

    def __collection_step(self):
        leader_state, follower_state = self.teleoperation_pair.follow()

        for cam_name, cam in self.cameras.items():
            obs = cam.get_observation()
            for stream_name, frame in obs.items():
                if stream_name in self.drop_stream_names:
                    continue

                if self.keep_stream_names and stream_name not in self.keep_stream_names:
                    continue

                # only resize and crop images
                if isinstance(frame, np.ndarray):
                    if (resize_wh := self.resize_wh[cam_name]) is not None:
                        if frame.dtype in (np.float32, np.float64):
                            # for depth
                            interpolation = cv2.INTER_NEAREST_EXACT
                        else:
                            # for RGB and IR
                            interpolation = cv2.INTER_LINEAR

                        frame = cv2.resize(
                            frame,  # type: ignore
                            resize_wh,
                            interpolation=interpolation,
                        )

                    if (crop_slice := self.crop_slices[cam_name]) is not None:
                        frame = frame[crop_slice]

                self.trajectory_cam_data[cam_name][stream_name].append(frame)

        self.trajectory_robot_data["leader"].append_state(leader_state)

        T_initial_ee = self.__pose_to_homogeneous_matrix(
            follower_state.ee_pos[:3], follower_state.ee_pos[3:]
        )

        self.trajectory_robot_data["follower"].append_state(
            follower_state, dynamic_extrinsics=T_initial_ee
        )

    def __pose_to_homogeneous_matrix(self, position, quaternion):
        """
        Convert a pose (position + quaternion) to a 4x4 homogeneous transformation matrix.

        Args:
            position (numpy.ndarray): 3D position (x, y, z).
            quaternion (numpy.ndarray): Quaternion (w, x, y, z).

        Returns:
            numpy.ndarray: 4x4 transformation matrix.
        """
        T = np.eye(4)  # Identity matrix
        T[:3, :3] = Rotation.from_quat(
            quaternion
        ).as_matrix()  # Convert quaternion to rotation matrix
        T[:3, 3] = position  # Set translation
        return T.astype(np.float32)

    def __save_data(self, counter):
        """
        Flush the buffered episode to episode.h5 with structure:

            obs/proprioception/{joint_pos,gripper_pos}
            obs/<cam>/{rgb,depth,intrinsics,extrinsics}

        """
        data_follower = self.trajectory_robot_data["follower"]
        data_leader = self.trajectory_robot_data["leader"]

        episode_td = TensorDict(
            {
                "obs": {
                    "proprioception": {
                        "joint_pos": torch.stack(data_follower.joint_pos_list),
                        "joint_vel": torch.stack(data_follower.joint_vel_list),
                        "gripper_pos": torch.stack(data_follower.gripper_state_list),
                        "eef_pos": torch.stack(data_follower.ee_pos_list),
                        "eef_quat": torch.stack(data_follower.ee_quat_list),
                        "eef_vel": torch.stack(data_follower.ee_vel_list),
                    },
                    **{
                        cam_name: {"frames": {}, "meta": {}}
                        for cam_name in self.cameras.keys()
                    },
                },
                "action": {
                    "joint_pos": torch.stack(data_leader.joint_pos_list),
                    "joint_vel": torch.stack(data_leader.joint_vel_list),
                    "gripper_pos": torch.stack(data_leader.gripper_state_list),  # [T]
                    "eef_pos": torch.stack(data_leader.ee_pos_list),
                    "eef_quat": torch.stack(data_leader.ee_quat_list),
                    "eef_vel": torch.stack(data_leader.ee_vel_list),  # [T, 6]
                },
            }  # type: ignore
        )

        for cam_name, cam_data in self.trajectory_cam_data.items():
            for stream_name in list(cam_data.keys()):
                # pop so that the original list can be garbage collected
                frames = cam_data.pop(stream_name)
                frames = np.stack(frames)

                # converts np to torch
                episode_td["obs", cam_name, "frames", stream_name] = frames

                del frames
                gc.collect()

            if cam_name == "gripper_cam":
                ee_pose = np.stack(data_follower.dynamic_extrinsics_list)
                episode_td["obs", cam_name, "frames", "dynamic_extrinsics"] = ee_pose

            episode_td["obs", cam_name, "meta"] = self.cam_calib[cam_name]

        h5_path = self.data_dir / f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.h5"

        print(f"Saving episode {counter} to {h5_path}")
        traj_length = len(data_follower.joint_pos_list)
        duration = time.perf_counter() - self.episode_start
        print(
            f"Trajectory stats:\n"
            f"Length (steps): {traj_length}\n"
            f"Duration (s): {duration:.2f}\n"
            f"Effective fps {traj_length / duration:.4f}"
        )

        episode_td.to_h5(
            str(h5_path),
            compression="gzip",
            compression_opts=7,
        )

    def __close_hardware_connections(self):
        self.teleoperation_pair.close()

        for cam in self.cameras.values():
            cam.close()
