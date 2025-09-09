import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Mapping, Optional

import h5py
import numpy as np
import pygame
import torch
from scipy.spatial.transform import Rotation
from tensordict import TensorDict

from environments.real_robot.hardware.base_camera import BaseCamera
from environments.real_robot.hardware.utils.keyboard import KeyManager
from environments.real_robot.teleop.teleoperation_base import (
    Robot,
    RobotState,
    TeleoperationPair,
)

ExtrinsicsMatrix = List[List[float]]


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
        baselines: Optional[dict[str, float]] = None,
        extrinsics: Optional[dict[str, ExtrinsicsMatrix]] = None,
        cam_name_mapping: dict[str, str] = None,
        cameras: Mapping[str, BaseCamera] | None = None,
        cam_keys: list[str] | None = None,
    ):
        self.teleoperation_pair = teleoperation_pair

        self.cameras = cameras or {}
        self.cam_keys = cam_keys
        self.cam_name_mapping = cam_name_mapping
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

        self.trajectory_robot_data: dict[str, CollectionData] = {}
        self.trajectory_cam_data: dict[str, dict[str, list]] = {}

        self.cam_calib: dict[str, dict] = {}
        for cam_name, cam in self.cameras.items():
            intr_dict = cam.get_intrinsics()
            fx = intr_dict["fx"]
            fy = intr_dict["fy"]
            cx = intr_dict["cx"]
            cy = intr_dict["cy"]
            intrinsics = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
            )
            height_width = np.array(
                [intr_dict["height"], intr_dict["width"]], dtype=np.int32
            )
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
                time.sleep(5)  # Give some time for the robots to reset

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

    def __reset_robots(self):
        self.teleoperation_pair.reset()

    def __collection_step(self):
        leader_state, follower_state = self.teleoperation_pair.follow()

        for cam_name, cam in self.cameras.items():
            obs = cam.get_observation()
            if self.cam_keys:
                obs = {k: obs[k] for k in self.cam_keys}

            for stream_name, frame in obs.items():
                if stream_name == "time":
                    continue
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

        proprio_td = {
            "joint_pos": torch.stack(data_follower.joint_pos_list),
            "joint_vel": torch.stack(data_follower.joint_vel_list),
            "gripper_pos": torch.stack(data_follower.gripper_state_list),
            "eef_pos": torch.stack(data_follower.ee_pos_list),
            "eef_quat": torch.stack(data_follower.ee_quat_list),
            "eef_vel": torch.stack(data_follower.ee_vel_list),
        }

        cam_tds = {
            cam_name: {"frames": {}, "meta": {}} for cam_name in self.cameras.keys()
        }
        for cam_name, cam_data in self.trajectory_cam_data.items():
            for stream_name in list(cam_data.keys()):
                # pop so that the original list can be garbage collected
                frames = cam_data.pop(stream_name)
                frames = np.stack(frames)
                cam_tds[cam_name]["frames"][stream_name] = frames

            if cam_name == "gripper_cam":
                ee_pose = np.stack(data_follower.dynamic_extrinsics_list)
                cam_tds[cam_name]["frames"]["dynamic_extrinsics"] = ee_pose

            cam_tds[cam_name]["meta"] = self.cam_calib[cam_name]

        obs_td = {"proprioception": proprio_td, **cam_tds}

        action_td = {
            "joint_pos": torch.stack(data_leader.joint_pos_list),
            "joint_vel": torch.stack(data_leader.joint_vel_list),
            "gripper_pos": torch.stack(data_leader.gripper_state_list),  # [T]
            "eef_pos": torch.stack(data_leader.ee_pos_list),
            "eef_quat": torch.stack(data_leader.ee_quat_list),
            "eef_vel": torch.stack(data_leader.ee_vel_list),  # [T, 6]
        }

        episode_td = TensorDict(
            {
                "obs": obs_td,
                "actions": action_td,
            },  # type: ignore
        )

        h5_path = self.data_dir / f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.h5"
        episode_td.to_h5(
            str(h5_path),
            compression="gzip",
            compression_opts=7,
        )

        print(f"Episode {counter} saved to {h5_path}")

    def __close_hardware_connections(self):
        self.teleoperation_pair.close()

        for cam in self.cameras.values():
            cam.close()
