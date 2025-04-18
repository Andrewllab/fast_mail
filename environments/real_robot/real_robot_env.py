import dataclasses
import logging
from typing import Mapping

import gymnasium as gym
import numpy as np
import torch

from environments.real_robot.hardware.hardware_cameras import DiscreteCamera
from environments.real_robot.hardware.hardware_franka import ControlType
from environments.real_robot.hardware.hardware_robot import RobotArm, RobotHand
from environments.specs import ActionSpec, DataSpecs, ObsSpec, specs_to_spaces
from utils.math import make_pose, quaternion_to_matrix

log = logging.getLogger(__name__)

# set initial pose!
DEFAULT_RESET_POSE = [
    0.0511338,
    -0.186824,
    -0.106011,
    -2.51019,
    -0.0180571,
    2.37377,
    0.739364,
]
CONTROL_TYPE = ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL_V2
# ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL_V2
GRIPPER_POS_SCALE = 0.04 / 0.07886763662099838


class RealRobotEnv(gym.Env):
    def __init__(
        self,
        robot_arm: RobotArm,
        robot_hand: RobotHand,
        cameras: Mapping[str, DiscreteCamera],
        use_delta: bool = True,
    ):
        self.use_delta = use_delta

        self.robot_arm = robot_arm(
            control_type=CONTROL_TYPE,
            default_reset_pose=DEFAULT_RESET_POSE,
        )
        self.robot_hand = robot_hand

        self.cameras = cameras
        for cam in cameras.values():
            assert isinstance(cam, DiscreteCamera), f"Invalid camera type: {type(cam)}"
            cam.connect()

        assert self.robot_arm.connect(), f"Connection to {self.robot_arm.name} failed"
        assert self.robot_hand.connect(), f"Connection to {self.robot_hand.name} failed"

        self.devices = [
            self.robot_arm,
            self.robot_hand,
        ] + list(self.cameras.values())

        # robot state
        # we concatenate joint_pos and gripper_pos to get a shape of (T, 9)
        robot_state = ObsSpec(elem_shape=(9,))

        # gripper_cam_transform
        gripper_cam_transform = ObsSpec(elem_shape=(4, 4))

        obs_specs = {
            "robot_state": robot_state,
            "gripper_cam_transform": gripper_cam_transform,
        }

        camera_specs = {key: camera.spec for key, camera in cameras.items()}

        # add the dynamic_pose_obs_key to the gripper_cam if we have it
        # this is needed for the complete extrinsics of the moving gripper cam
        if "gripper_cam" in camera_specs:
            camera_specs["gripper_cam"] = dataclasses.replace(
                camera_specs["gripper_cam"],
                dynamic_pose_obs_key="gripper_cam_transform",
            )

        obs_specs.update(camera_specs)

        # actions
        action = ActionSpec(action_dim=8)

        self._specs = DataSpecs(
            obs=obs_specs,
            action=action,
        )

        self.observation_space, self.action_space = specs_to_spaces(self._specs)

        # self.ee_positions = []
        # self.ee_quaternions = []

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def step(self, action_np: np.ndarray):

        action_np /= 1.0  # reverse scaling applied in recording demos

        # log.debug(f"Action: {action_np}")

        action = torch.from_numpy(action_np)
        wxyz = action[3:7]
        xyzw = torch.cat((wxyz[-3:], wxyz[:-3]), dim=0)

        ee_pos, ee_quat = self.robot_arm.apply_ee(
            position=action[:3],
            orientation=xyzw,
            delta=self.use_delta,
        )

        self.robot_hand.apply_commands(action[-1])

        obs = self._get_obs()
        info = self._get_info()
        info["current_ee_pos"] = ee_pos
        info["current_ee_rot"] = ee_quat

        return obs, 0, False, False, info

    def reset(self, *, seed=None, options=None):
        self.robot_arm.reset()
        self.robot_hand.reset()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def close(self):
        for device in self.devices:
            if not device.close():
                log.warning(f"Failed to close {device.name}")

    def _get_obs(self):
        gripper_width = self.robot_hand.get_sensors() * GRIPPER_POS_SCALE
        robot_state = np.concatenate(
            (
                self.robot_arm.get_state().joint_pos.numpy(),  # 7
                gripper_width,
                -gripper_width,
            ),
            axis=-1,
        )

        robot_arm_ee_pose = self.robot_arm.get_state().ee_pos

        pos, xyzw = robot_arm_ee_pose[:3], robot_arm_ee_pose[3:]
        wxyz = torch.cat((xyzw[3:], xyzw[:3]), dim=0)
        rot = quaternion_to_matrix(wxyz)
        gripper_cam_transform = make_pose(pos, rot).numpy()

        obs_dict = {
            "robot_state": robot_state,
            "gripper_cam_transform": gripper_cam_transform,
        }

        images = {key: camera.get_sensors() for key, camera in self.cameras.items()}

        obs_dict.update(images)

        return obs_dict

    def _get_info(self):
        return {}
