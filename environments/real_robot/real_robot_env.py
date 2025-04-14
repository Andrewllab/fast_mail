import logging
from typing import Mapping

import gymnasium as gym
import numpy as np
import torch

from environments.real_robot.hardware.hardware_cameras import DiscreteCamera
from environments.real_robot.hardware.hardware_franka import ControlType
from environments.real_robot.hardware.hardware_robot import RobotArm, RobotHand
from environments.specs import (
    ActionSpec,
    DataSpecs,
    PinholeCameraIntrinsic,
    RGBCameraSpec,
    Spec,
    specs_to_spaces,
)

# from utils.math import euler_xyz_to_quaternion

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
            # default_reset_pose=DEFAULT_RESET_POSE,
        )
        self.robot_hand = robot_hand

        self.left_cam = cameras["left_cam"]
        self.left_cam.connect()
        self.right_cam = cameras["right_cam"]
        self.right_cam.connect()
        self.gripper_cam = cameras["gripper_cam"]
        self.gripper_cam.connect()

        assert self.robot_arm.connect(), f"Connection to {self.robot_arm.name} failed"
        assert self.robot_hand.connect(), f"Connection to {self.robot_hand.name} failed"

        self.devices = [
            self.robot_arm,
            self.robot_hand,
            self.gripper_cam,
            self.left_cam,
            self.right_cam,
        ]

        # static camera front left
        left_cam = RGBCameraSpec(
            shape=(480, 640, 3),
            # intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
            #     intrinsics, width=shape[2], height=shape[1]
            # ),
            # orthogonal=False,
            channel_order="HWC",
        )

        # static camera front right
        right_cam = RGBCameraSpec(
            shape=(480, 640, 3),
            # intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
            #     intrinsics, width=shape[2], height=shape[1]
            # ),
            # orthogonal=False,
            channel_order="HWC",
        )

        # gripper camera
        gripper_cam = RGBCameraSpec(
            shape=(480, 640, 3),
            # intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
            #     intrinsics, width=shape[2], height=shape[1]
            # ),
            # dynamic_pose_obs_key="gripper_cam_transform",
            # extrinsics=None,  # gripper_cam_transform provides complete transform
            # orthogonal=False,
            channel_order="HWC",
        )

        # robot state
        # we concatenate joint_pos and gripper_pos to get a shape of (T, 9)
        robot_state = Spec(shape=(9,), type="state")

        # gripper_cam_transform
        gripper_cam_transform = Spec(shape=(4, 4), type="transform")

        # actions
        action = ActionSpec(shape=(8,), type="action")

        self._specs = DataSpecs(
            obs={
                "left_cam": left_cam,
                "right_cam": right_cam,
                "gripper_cam": gripper_cam,
                "robot_state": robot_state,
                # "gripper_cam_transform": gripper_cam_transform,
            },
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

        log.debug(f"Action: {action_np}")

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

        obs_dict = {
            "robot_state": robot_state,
            "left_cam": self.left_cam.get_sensors()["rgb"],
            "right_cam": self.right_cam.get_sensors()["rgb"],
            "gripper_cam": self.gripper_cam.get_sensors()["rgb"],
        }

        return obs_dict

    def _get_info(self):
        return {}
