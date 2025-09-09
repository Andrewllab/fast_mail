import logging
from enum import Enum, auto
from typing import  Literal
from dataclasses import dataclass

import torch
from polymetis import RobotInterface, GripperInterface

logger = logging.getLogger(__name__)

class TeleoperationType(Enum):
    JOINT_SPACE = auto()
    TASK_SPACE = auto()

@dataclass
class RobotState:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    ee_pos: torch.Tensor
    ee_vel: torch.Tensor
    gripper_state: torch.Tensor


class Robot:
    def __init__(self, name: str, ip_address: str, arm_port: int, gripper_port: int):
        self.name = name
        self.robot_arm = RobotInterface(
            name=name, ip_address=ip_address, port=arm_port, enforce_version=False
        )
        logger.info(f"Connected to {name}'s robot arm at {ip_address}:{arm_port}")

        self.robot_gripper = GripperInterface(ip_address, gripper_port)
        logger.info(f"Connected to {name}'s robot gripper at {ip_address}:{gripper_port}")

    def close(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class TeleoperationPair:

    @property
    def leader_robot(self) -> Robot:
        raise NotImplementedError

    @property
    def follower_robot(self) -> Robot:
        raise NotImplementedError

    @property
    def teleoperation_type(self) -> TeleoperationType:
        raise NotImplementedError

    def follow(self) -> tuple[RobotState, RobotState]:
        raise NotImplementedError

    def close(self):
        self.follower_robot.close()
        self.leader_robot.close()

    def reset(self):
        self.follower_robot.reset()
        self.leader_robot.reset()