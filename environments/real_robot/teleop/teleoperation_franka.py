import logging
import time
from typing import Any, Literal, Optional, Sequence

import torch
from torchcontrol.policies import (
    CartesianImpedanceControl,
    HybridJointImpedanceControl,
    JointImpedanceControl,
)

from environments.real_robot.hardware.franka_control import HumanControl
from environments.real_robot.teleop.teleoperation_base import (
    Robot,
    RobotState,
    TeleoperationPair,
)

logger = logging.getLogger(__name__)


class FrankaRobot(Robot):
    def __init__(
        self,
        name: str,
        ip_address: str,
        arm_port: int,
        gripper_port: int,
        control_type: Literal["joint", "cartesian", "hybrid_joint", "human"],
        home_pose: Sequence[float] | None = None,
        gripper_speed: float = 0.1,
        gripper_force: float = 0.1,
        binary_gripper_state: bool = False,
    ):
        super().__init__(name, ip_address, arm_port, gripper_port)

        self._home_pose = home_pose
        if self._home_pose is not None:
            logger.info(f"Setting home pose: {self._home_pose}")
            self.robot_arm.set_home_pose(torch.tensor(self._home_pose))

        self.gripper_speed = gripper_speed
        self.gripper_force = gripper_force
        self.gripper_max_width = self.robot_gripper.metadata.max_width
        logger.debug(f"Gripper speed: {self.gripper_speed}")
        logger.debug(f"Gripper force: {self.gripper_force}")
        logger.debug(f"Gripper max width: {self.gripper_max_width}")

        assert self.gripper_max_width > 0, "Gripper max width must be greater than 0"

        self.control_type = control_type
        self.binary_gripper_state = binary_gripper_state

    def send_policy(self):
        if self.control_type == "human":
            policy = HumanControl(self.robot_arm, regularize=True)
        elif self.control_type == "cartesian":
            policy = CartesianImpedanceControl(
                joint_pos_current=self.robot_arm.get_joint_positions(),
                Kp=self.robot_arm.Kx_default,
                Kd=self.robot_arm.Kxd_default,
                robot_model=self.robot_arm.robot_model,
                ignore_gravity=self.robot_arm.use_grav_comp,
            )
        elif self.control_type == "hybrid_joint":
            policy = HybridJointImpedanceControl(
                joint_pos_current=self.robot_arm.get_joint_positions(),
                Kq=self.robot_arm.Kq_default,
                Kqd=self.robot_arm.Kqd_default,
                Kx=self.robot_arm.Kx_default,
                Kxd=self.robot_arm.Kxd_default,
                robot_model=self.robot_arm.robot_model,
                ignore_gravity=self.robot_arm.use_grav_comp,
            )
        elif self.control_type == "joint":
            policy = JointImpedanceControl(
                joint_pos_current=self.robot_arm.get_joint_positions(),
                Kp=self.robot_arm.Kq_default,
                Kd=self.robot_arm.Kqd_default,
                robot_model=self.robot_arm.robot_model,
                ignore_gravity=self.robot_arm.use_grav_comp,
            )
        else:
            raise ValueError(f"Unknown control type: {self.control_type}")

        self.robot_arm.send_torch_policy(policy, blocking=False)

    def close(self):
        self.robot_arm.terminate_current_policy()

    def reset(self):
        self.robot_gripper.goto(
            self.gripper_max_width,
            speed=self.gripper_speed,
            force=self.gripper_force,
            blocking=False,
        )
        self.robot_arm.go_home()

        self.robot_gripper.grasp(
            self.gripper_speed,
            force=self.gripper_force,
            blocking=False,
        )
        time.sleep(1.0)
        self.robot_gripper.goto(
            self.gripper_max_width,
            speed=self.gripper_speed,
            force=self.gripper_force,
            blocking=False,
        )

        self.send_policy()

    def get_state(self) -> RobotState:
        arm_state = self.robot_arm.get_robot_state()
        joint_pos = torch.tensor(arm_state.joint_positions)
        joint_vel = torch.tensor(arm_state.joint_velocities)

        pos, quat = self.robot_arm.robot_model.forward_kinematics(joint_pos)
        ee_pos = torch.cat([pos, quat])

        jacobian = self.robot_arm.robot_model.compute_jacobian(joint_pos)
        ee_vel = jacobian @ joint_vel

        gripper_width = self.robot_gripper.get_state().width
        if self.binary_gripper_state:
            thresh = self.gripper_max_width / 2
            gripper_state = torch.tensor(-1 if gripper_width < thresh else 1)
        else:
            gripper_state = torch.tensor(gripper_width)

        return RobotState(
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            ee_pos=ee_pos,
            ee_vel=ee_vel,
            gripper_state=gripper_state,
        )


class FrankaTeleoperationPair(TeleoperationPair):

    def __init__(
        self,
        leader_ip_address: str,
        leader_ports: tuple[int, int],
        follower_ip_address: str,
        follower_ports: tuple[int, int],
        control_type: Literal["joint", "cartesian", "hybrid_joint"],
        home_pose: Optional[Any] = None,
    ):
        self._leader_robot = FrankaRobot(
            name="Leader Franka",
            ip_address=leader_ip_address,
            arm_port=leader_ports[0],
            gripper_port=leader_ports[1],
            control_type="human",
            home_pose=home_pose,
            # we want to observe the value we are tracking, not the raw width
            binary_gripper_state=True,
        )

        self._control_type = control_type
        self._follower_robot = FrankaRobot(
            name="Follower Franka",
            ip_address=follower_ip_address,
            arm_port=follower_ports[0],
            gripper_port=follower_ports[1],
            control_type=control_type,
            home_pose=home_pose,
        )

    @property
    def leader_robot(self) -> Robot:
        return self._leader_robot

    @property
    def follower_robot(self) -> Robot:
        return self._follower_robot

    @property
    def control_type(self) -> Literal["joint", "cartesian", "hybrid_joint"]:
        return self._control_type

    def follow(self) -> tuple[RobotState, RobotState]:
        leader_state = self._leader_robot.get_state()
        follower_state = self._follower_robot.get_state()

        # update target positions and velocities of follower robot
        if self._control_type in ["joint", "hybrid_joint"]:
            self.follower_robot.robot_arm.update_current_policy(
                {
                    "joint_pos_desired": leader_state.joint_pos,
                    # "joint_vel_desired": leader_state.joint_vel,
                }
            )
        elif self._control_type == "cartesian":
            self.follower_robot.robot_arm.update_current_policy(
                {
                    "ee_pos_desired": leader_state.ee_pos[:3],
                    "ee_quat_desired": leader_state.ee_pos[3:],
                    "ee_vel_desired": leader_state.ee_vel[:3],
                    "ee_rvel_desired": leader_state.ee_vel[3:],
                }
            )
        else:
            raise ValueError(f"Unknown control type: {self._control_type}")

        # set target gripper state of follower robot
        self.follower_robot.robot_gripper.set_state(
            leader_state.gripper_state,
            speed=self._follower_robot.gripper_speed,
            force=self._follower_robot.gripper_force,
        )

        return leader_state, follower_state
