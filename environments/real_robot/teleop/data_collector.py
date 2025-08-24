import time
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

import pygame
import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tensordict import TensorDict

from environments.real_robot.hardware.base_camera import BaseCamera
from environments.real_robot.hardware.utils.keyboard import KeyManager
from environments.real_robot.teleop.teleoperation_base import (
    TeleoperationPair,
    Robot,
    RobotState,
)

ExtrinsicsMatrix = List[List[float]]


class CollectionData:
    joint_pos_list: List[torch.Tensor]
    joint_vel_list: List[torch.Tensor]
    ee_pos_list: List[torch.Tensor]
    ee_quat_list: List[torch.Tensor]
    ee_vel_list: List[torch.Tensor]
    gripper_state_list: List[Literal[-1, 1]]
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
        cameras: list[BaseCamera] = [],
        cam_keys: list[str] | None = None,
    ):
        self.teleoperation_pair = teleoperation_pair

        self.cameras = cameras
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
        self.data_dir.mkdir(exist_ok=True)

        self.clock = pygame.time.Clock()
        self.step_hz = step_hz

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
                        counter += 1
                        print("COUNTER", counter)

                        print("Saved!")
                    elif km.key == "d":
                        print()
                        print("Discarding collected data")
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
        """Allocate per‑robot trajectory buffers and per‑camera frame buffers."""
        self.robot_to_data: dict[Robot, CollectionData] = {
            self.teleoperation_pair.leader_robot: CollectionData(),
            self.teleoperation_pair.follower_robot: CollectionData(),
        }

        self.cam_episode_buffer: dict[str, list[dict]] = {
            cam.name: [] for cam in self.cameras
        }

        self.cam_calib: dict[str, dict] = {}

        for cam in self.cameras:
            intr_dict = cam.get_intrinsics()
            fx = intr_dict["fx"]
            fy = intr_dict["fy"]
            cx = intr_dict["cx"]
            cy = intr_dict["cy"]
            intrinsics = np.array(
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
            )

            baseline = (
                self.baselines[cam.name]
                if cam.name in self.baselines
                else intr_dict.get("baseline", None)
            )

            generic_name = self.cam_name_mapping[cam.name]
            extrinsics = self.extrinsics[generic_name] if self.extrinsics else None

            self.cam_calib[generic_name] = {
                "intrinsics": intrinsics,
                "intrinsics_heights_width": np.array(
                    [intr_dict["height"], intr_dict["width"]], dtype=np.int32
                ),
                "baseline": baseline,
                "extrinsics": extrinsics,
            }

    def __reset_robots(self):
        self.teleoperation_pair.reset()

    def __collection_step(self):
        leader_state, follower_state = self.teleoperation_pair.follow()

        for cam in self.cameras:
            obs = cam.get_observation()
            if self.cam_keys:
                obs = {k: obs[k] for k in self.cam_keys}
            self.cam_episode_buffer[cam.name].append(obs)

        self.robot_to_data[self.teleoperation_pair.leader_robot].append_state(
            leader_state
        )

        T_initial_ee = self.__pose_to_homogeneous_matrix(
            follower_state.ee_pos[:3], follower_state.ee_pos[3:]
        )

        self.robot_to_data[self.teleoperation_pair.follower_robot].append_state(
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
        return T

    def __save_data(self, counter):
        """
        Flush the buffered episode to episode.h5 with structure:

            obs/proprioception/{joint_pos,gripper_pos}
            obs/<cam>/{rgb,depth,intrinsics,extrinsics}

        """
        data_follower = self.robot_to_data[self.teleoperation_pair.follower_robot]
        data_leader = self.robot_to_data[self.teleoperation_pair.leader_robot]

        T = len(data_follower.joint_pos_list)

        proprio_td = TensorDict(
            {
                "joint_pos": torch.stack(data_follower.joint_pos_list),
                "joint_vel": torch.stack(data_follower.joint_vel_list),
                "gripper_pos": torch.tensor(data_follower.gripper_state_list),
                "eef_pos": torch.stack(data_follower.ee_pos_list),
                "eef_quat": torch.stack(data_follower.ee_quat_list),
                "eef_vel": torch.stack(data_follower.ee_vel_list),
            },
            batch_size=[T],
        )
        cam_tds: dict[str, TensorDict] = {}
        for original_name, frames in self.cam_episode_buffer.items():
            if not frames:
                continue

            assert (
                original_name in self.cam_name_mapping
            ), f"Unknown camera name: {original_name}"
            generic_name = self.cam_name_mapping[original_name]

            first = frames[0]
            dynamic_fields = {}
            for k in first.keys():
                stacked = np.stack([f[k] for f in frames], axis=0)
                dynamic_fields[k] = torch.from_numpy(stacked)

            if generic_name == "gripper_cam":
                dynamic_fields["dynamic_extrinsics"] = torch.stack(
                    [
                        torch.from_numpy(arr)
                        for arr in data_follower.dynamic_extrinsics_list
                    ]
                )

            cam_dynamic_td = TensorDict(dynamic_fields, batch_size=[T])

            cam_tds[generic_name] = TensorDict(
                {
                    "frames": cam_dynamic_td,
                },
                batch_size=[T],
            )

        obs_td = TensorDict(
            {"proprioception": proprio_td, **cam_tds},
            batch_size=[T],
        )

        action_td = TensorDict(
            {
                "joint_pos": torch.stack(data_leader.joint_pos_list),
                "joint_vel": torch.stack(data_leader.joint_vel_list),
                "gripper_pos": torch.tensor(data_leader.gripper_state_list),  # [T]
                "eef_pos": torch.stack(data_leader.ee_pos_list),
                "eef_quat": torch.stack(data_leader.ee_quat_list),
                "eef_vel": torch.stack(data_leader.ee_vel_list),  # [T, 6]
            },
            batch_size=[T],
        )

        episode_td = TensorDict(
            {
                "obs": obs_td,
                "actions": action_td,
            },
            batch_size=[T],
        )

        h5_path = self.data_dir / f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.h5"
        episode_td.to_h5(
            str(h5_path),
            compression="gzip",
            compression_opts=7,
        )

        with h5py.File(str(h5_path), "a") as f:
            for cam_name, _ in cam_tds.items():
                meta_group = f.create_group(f"obs/{cam_name}/meta")
                meta = self.cam_calib[cam_name]
                for k, v in meta.items():
                    if v is None:
                        continue
                    data = v.numpy() if isinstance(v, torch.Tensor) else v
                    meta_group.create_dataset(k, data=data)

        print(f"Episode {counter} saved to {h5_path}")

    def __close_hardware_connections(self):
        self.teleoperation_pair.close()

        for cam in self.cameras:
            cam.close()
