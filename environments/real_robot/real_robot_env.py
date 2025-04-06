import gymnasium as gym

import torch
from tensordict import TensorDict
import numpy as np

from environments.real_robot.hardware.hardware_devices import (
    ContinuousDevice,
    DiscreteDevice,
)
from environments.real_robot.hardware.hardware_franka import ControlType
from environments.real_robot.hardware.hardware_robot import RobotArm, RobotHand
from environments.specs import (
    ActionSpec,
    DataSpecs,
    PinholeCameraIntrinsic,
    RGBDCameraSpec,
    Spec,
)
from utils.math import euler_xyz_to_quaternion


class RealRobotEnv(gym.Env):
    def __init__(
        self,
        robot_arm: RobotArm,
        robot_hand: RobotHand,
        discrete_devices: list[DiscreteDevice] | None = None,
        continuous_devices: list[ContinuousDevice] | None = None,
    ):
        self.robot_arm = robot_arm(control_type=ControlType.CARTESIAN_IMPEDANCE_CONTROL)
        self.robot_hand = robot_hand
        self.discrete_devices = discrete_devices or []
        self.continuous_devices = continuous_devices or []

        assert self.robot_arm.connect(), f"Connection to {self.robot_arm.name} failed"
        assert self.robot_hand.connect(), f"Connection to {self.robot_hand.name} failed"

        for device in self.discrete_devices:
            assert device.connect(), f"Connection to {device.name} failed"

        for device in self.continuous_devices:
            assert device.connect(), f"Connection to {device.name} failed"

        # static camera front left
        left_cam = RGBDCameraSpec(
            shape=(1, 480, 640, 3),
            # intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
            #     intrinsics, width=shape[2], height=shape[1]
            # ),
            orthogonal=False,
            channel_order="HWC",
        )

        # static camera front right
        right_cam = RGBDCameraSpec(
            shape=(1, 480, 640, 3),
            # intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
            #     intrinsics, width=shape[2], height=shape[1]
            # ),
            orthogonal=False,
            channel_order="HWC",
        )

        # gripper camera
        gripper_cam = RGBDCameraSpec(
            shape=(1, 480, 640, 3),
            # intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
            #     intrinsics, width=shape[2], height=shape[1]
            # ),
            # dynamic_pose_obs_key="gripper_cam_transform",
            # extrinsics=None,  # gripper_cam_transform provides complete transform
            orthogonal=False,
            channel_order="HWC",
        )

        # robot state
        # we concatenate joint_pos and gripper_pos to get a shape of (T, 9)
        robot_state = Spec(shape=(1, 9), type="state")

        # gripper_cam_transform
        gripper_cam_transform = Spec(shape=(1, 4, 4), type="transform")

        # actions
        action = ActionSpec(
            shape=(1, 7), type="action"
        )

        self._specs = DataSpecs(
            obs={
                "left_cam": left_cam,
                "right_cam": right_cam,
                "gripper_cam": gripper_cam,
                "robot_state": robot_state,
                "gripper_cam_transform": gripper_cam_transform,
            },
            action=action,
        )

        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(1, 7), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
                # "camera": gym.spaces.Box(
                #     low=0, high=255, shape=(1, 480, 640, 3), dtype=np.uint8
                # ),
                "robot_state": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1, 8), dtype=np.float32
                ),
            }
        )

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def step(self, action: np.ndarray):
        euler = action[3:6]
        wxyz = euler_xyz_to_quaternion(*torch.from_numpy(euler))
        poly_action = np.concat((action[:3], wxyz[1:], wxyz[:1]))

        self.robot_arm.go_to_within_limits(poly_action)
        self.robot_hand.apply_commands(action[-1])

        obs = self._get_obs()
        info = self._get_info()

        return obs, 0, False, False, info

    def reset(self, *, seed=None, options=None):
        for device in self.continuous_devices:
            device.stop_recording()

        self.robot_arm.reset()
        self.robot_hand.reset()

        for device in self.continuous_devices:
            device.start_recording()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def close(self):
        for device in self.continuous_devices:
            device.stop_recording()

        if not self.robot_arm.close():
            print(f"Failed to close {self.robot_arm.name}")

        if not self.robot_hand.close():
            print(f"Failed to close {self.robot_hand.name}")

        for device in self.discrete_devices:
            if not device.close():
                print(f"Failed to close {device.name}")

        for device in self.continuous_devices:
            if not device.close():
                print(f"Failed to close {device.name}")

    def _get_obs(self):
        obs_dict = {}

        robot_state = np.concatenate(
            (
                self.robot_arm.get_state().joint_pos.numpy(), # TODO: Update the input # shape: (T, 9) if 9, 0,1,7 is not important!
                self.robot_hand.get_sensors(),  # TODO: check get_sensors output and compare it with dataset
            ),
            axis=-1,
        )

        robot_state = robot_state[None, ...]

        # TODO: will probably need to pop the timestamp from the obs
        for device in self.discrete_devices:
            obs_dict[device.name] = device.get_sensors() # TODO: Pass everything or just the necessary things? {"time": timestamp, "rgb": rgb, "d": d, "ir1": ir1, "ir2": ir2}

        # traj = TensorDict(
        #     {
        #         "obs": {
        #             # "left_cam": obs_dict["RealSense_243322073029"], # TODO Get serial number
        #             # "right_cam": obs_dict["RealSense_944622073668"], # TODO Get serial number
        #             # "gripper_cam": obs_dict["RealSense_218622270040"], # TODO Get serial number
        #             "robot_state": robot_state,
        #         },
        #     },  # type: ignore
        # )

        # # add batch dimension
        # traj._unsqueeze(0)

        # # not necessary because we don't have batches in inference
        # traj.auto_batch_size_(batch_dims=1)

        return {
            # "left_cam": obs_dict["RealSense_243322073029"], # TODO Get serial number
            # "right_cam": obs_dict["RealSense_944622073668"], # TODO Get serial number
            # "gripper_cam": obs_dict["RealSense_218622270040"], # TODO Get serial number
            "robot_state": robot_state,
        }



    def _get_info(self):
        return {}
