import gymnasium as gym

import torch
from tensordict import TensorDict

from environments.real_robot.hardware.hardware_devices import (
    ContinuousDevice,
    DiscreteDevice,
)
from environments.real_robot.hardware.hardware_robot import RobotArm, RobotHand


class RealRobotEnv(gym.Env):
    def __init__(
        self,
        robot_arm: RobotArm,
        robot_hand: RobotHand,
        discrete_devices: list[DiscreteDevice] | None = None,
        continuous_devices: list[ContinuousDevice] | None = None,
    ):
        self.robot_arm = robot_arm
        self.robot_hand = robot_hand
        self.discrete_devices = discrete_devices or []
        self.continuous_devices = continuous_devices or []

        assert robot_arm.connect(), f"Connection to {robot_arm.name} failed"
        assert robot_hand.connect(), f"Connection to {robot_hand.name} failed"

        for device in self.discrete_devices:
            assert device.connect(), f"Connection to {device.name} failed"

        for device in self.continuous_devices:
            assert device.connect(), f"Connection to {device.name} failed"

        # TODO: add action and observation spaces

    def step(self, action: dict):
        # TODO: action cannot be a dictionary
        self.robot_arm.go_to_within_limits(action=action["robot_arm"])
        self.robot_hand.apply_commands(action=action["robot_hand"])

        obs = self._get_obs()
        info = self._get_info()

        # TODO: return tuple should have 5 elements
        return obs, 0, False, False, info, False

    def reset(self):
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

        robot_state = torch.cat(
            (
                self.robot_arm.get_state().joint_pos, # TODO: Update the input # shape: (T, 9) if 9, 0,1,7 is not important!
                self.robot_hand.get_sensors(),  # TODO: check get_sensors output and compare it with dataset
            ),
            dim=-1,
        )

        # TODO: will probably need to pop the timestamp from the obs
        for device in self.discrete_devices:
            obs_dict[device.name] = device.get_sensors() # TODO: Pass everything or just the necessary things? {"time": timestamp, "rgb": rgb, "d": d, "ir1": ir1, "ir2": ir2}

        action = torch.zeros(11) # TODO: Necessary?

        traj = TensorDict(
            {
                "obs": {
                    "front_left_cam": obs_dict["RealSense_X"], # TODO Get serial number
                    "front_right_cam": obs_dict["RealSense_Y"], # TODO Get serial number
                    "gripper_cam": obs_dict["RealSense_22xx"], # TODO Get serial number
                    "robot_state": robot_state,
                },
                "action": action, # TODO
            },  # type: ignore
        )

        # not necessary because we don't have batches in inference
        #traj.auto_batch_size_()

        return traj



    def _get_info(self):
        return {}
