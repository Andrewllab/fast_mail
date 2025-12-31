import dataclasses
import logging
import time
from typing import Literal, Mapping

import gymnasium as gym
import pygame
import torch
from omegaconf import DictConfig
from polymetis import GripperInterface, RobotInterface
from torchcontrol.policies import (
    CartesianImpedanceControl,
    HybridJointImpedanceControl,
    JointImpedanceControl,
)

from environments.real_robot.hardware.base_camera import BaseCamera
from environments.real_robot.hardware.franka_control import HumanControl
from environments.signals import DoneEvalSignal
from environments.specs import ActionSpec, DataSpecs, ObsSpec, specs_to_spaces
from utils.math import make_pose, normalize, quaternion_to_matrix

ObsType = dict[str, torch.Tensor | dict[str, torch.Tensor]]
InfoType = dict[str, torch.Tensor]

log = logging.getLogger(__name__)


class RealRobotEnv(gym.Env):
    # we don't implement a render method, but the obs contains camera images
    render_mode = "rgb_array"

    def __init__(
        self,
        robot: DictConfig,
        cameras: Mapping[str, BaseCamera] | None = None,
        action_type: Literal["joint", "cartesian"] = "joint",
        impedance_type: Literal["joint", "cartesian", "hybrid_joint"] = "hybrid_joint",
        human_control: bool = False,
    ):
        if action_type not in ("joint", "cartesian"):
            raise ValueError('action_type must be either "joint" or "cartesian"')
        if impedance_type not in ("joint", "cartesian", "hybrid_joint"):
            raise ValueError(
                'impedance_type must be either "joint", "cartesian", or "hybrid_joint"'
            )
        if action_type == "joint" and impedance_type not in ("joint", "hybrid_joint"):
            # in other words, cartesian impedance control only works with cartesian actions
            raise ValueError(
                "Joint space actions require joint impedance or hybrid joint impedance control."
            )
        self.action_type = action_type
        self.impedance_type = impedance_type
        self.human_control = human_control

        self.arm = RobotInterface(
            name=robot.name,
            ip_address=robot.ip_address,
            port=robot.arm_port,
            enforce_version=False,
        )
        log.info(
            f'Connected to "{robot.name}" robot arm at {robot.ip_address}:{robot.arm_port}'
        )

        self.gripper = GripperInterface(
            ip_address=robot.ip_address,
            port=robot.gripper_port,
        )
        log.info(
            f'Connected to "{robot.name}" gripper at {robot.ip_address}:{robot.gripper_port}'
        )

        self.gripper_speed = robot.get("gripper_speed", 0.1)
        self.gripper_force = robot.get("gripper_force", 0.1)
        self.gripper_max_width = self.gripper.metadata.max_width
        log.debug(f"Gripper max width: {self.gripper_max_width}")

        if (home_pose := robot.get("home_pose", None)) is not None:
            log.info(f"Setting home pose: {home_pose}")
            self.arm.set_home_pose(torch.tensor(home_pose))

        # default (None) uses _adaptive_time_to_go
        self.reset_duration = robot.get("reset_duration", 1.0)

        self.cameras: Mapping[str, BaseCamera] = cameras or {}

        obs_specs = {
            # joint pos (7,) + gripper_width (1,)
            "robot_state": (ObsSpec(elem_shape=(8,))),
            # xyz + wxyz quaternion
            "ee_pose": ObsSpec(elem_shape=(7,)),
            # homogeneous transform of ee_pose
            "ee_transform": ObsSpec(elem_shape=(4, 4)),
            "target_gripper_pos": ObsSpec(elem_shape=(1,)),
        }

        # TODO: Add both specs in either case.
        if action_type == "joint":
            obs_specs["target_joint_pos"] = ObsSpec(elem_shape=(7,))
        elif action_type == "cartesian":
            obs_specs["target_ee_pose"] = ObsSpec(elem_shape=(7,))

        camera_specs = {key: camera.spec for key, camera in self.cameras.items()}

        # add the dynamic_pose_obs_key to the gripper_cam if we have it
        # this is needed for the complete extrinsics of the moving gripper cam
        if "gripper_cam" in camera_specs:
            camera_specs["gripper_cam"] = dataclasses.replace(
                camera_specs["gripper_cam"],
                dynamic_pose_obs_key="ee_transform",
            )

        obs_specs.update(camera_specs)

        # target joint position (7,) or target ee pose (7,) + target gripper width (1,)
        action = ActionSpec(action_dim=8)

        self._specs = DataSpecs(
            obs=obs_specs,
            action=action,
        )

        self.observation_space, self.action_space = specs_to_spaces(self._specs)

        # Tiny, borderless window so we can capture keys (must be focused at least once)
        pygame.init()
        pygame.display.set_mode((1, 1), pygame.NOFRAME)
        pygame.display.set_caption(
            "Robot Control - Press Esc/q for failure, Enter for success, Space to pause"
        )
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])
        log.warning("Press Esc/q for failure, Enter for success, Space to pause")

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def step(self, action: torch.Tensor) -> tuple[ObsType, float, bool, bool, InfoType]:
        action = action.cpu()

        if self.action_type == "cartesian":
            target_ee_pose = action[:7]  # xyz + wxyz quaternion
            target_ee_pos = target_ee_pose[:3]
            target_ee_wxyz = target_ee_pose[3:7]
            # normalize quaternions coming from outside to ensure they are valid
            # BackCompat: this can be probably be removed because it is done
            # in the QuaternionRotations transform
            target_ee_wxyz = normalize(target_ee_wxyz)
            target_ee_xyzw = torch.cat(
                (target_ee_wxyz[-3:], target_ee_wxyz[:-3]), dim=-1
            )
        elif self.action_type == "joint":
            target_joint_pos = action[:7]

        if self.human_control:
            # in human control mode, we do not execute the actions from the policy
            pass
        elif self.impedance_type == "cartesian":
            self.arm.update_current_policy(
                {"ee_pos_desired": target_ee_pos, "ee_quat_desired": target_ee_xyzw}
            )
        # joint and hybrid joint impedance controllers
        elif self.action_type == "cartesian":
            self.arm.update_desired_ee_pose(
                position=target_ee_pos, orientation=target_ee_xyzw
            )
        elif self.action_type == "joint":
            self.arm.update_desired_joint_positions(positions=target_joint_pos)

        target_gripper_state = action[7]
        if not self.human_control:
            self.gripper.set_state(
                target_gripper_state.item(),
                speed=self.gripper_speed,
                force=self.gripper_force,
            )

        obs = self._get_obs()
        if self.action_type == "joint":
            obs["target_joint_pos"] = target_joint_pos
        elif self.action_type == "cartesian":
            obs["target_ee_pose"] = target_ee_pose
        obs["target_gripper_pos"] = target_gripper_state.unsqueeze(dim=0)

        info = self._get_info()
        result = self._get_user_input()

        if result == "success":
            reward, terminated, truncated = 1.0, True, False
        elif result == "fail":
            reward, terminated, truncated = 0.0, False, True
        else:
            reward, terminated, truncated = 0.0, False, False

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None) -> tuple[ObsType, InfoType]:
        # open gripper and go home simultaneously
        self.gripper.goto(self.gripper_max_width, speed=self.gripper_speed)

        options = options or {}
        if "home_pose" in options:
            self.arm.set_home_pose(options["home_pose"])
        elif "home_position" in options and "home_orientation" in options:
            self.arm.set_home_ee_pose(
                home_position=options["home_position"],
                home_orientation=options["home_orientation"],
            )

        # wait for the arm to go home
        self.arm.go_home(time_to_go=self.reset_duration)

        # wait a little longer in case the gripper is still moving
        time.sleep(1.0)

        # start the continuouos control policy
        if self.human_control:
            policy = HumanControl(self.arm, regularize=True)
        elif self.impedance_type == "cartesian":
            policy = CartesianImpedanceControl(
                joint_pos_current=self.arm.get_joint_positions(),
                Kp=self.arm.Kx_default,
                Kd=self.arm.Kxd_default,
                robot_model=self.arm.robot_model,
                ignore_gravity=self.arm.use_grav_comp,
            )
        elif self.impedance_type == "hybrid_joint":
            policy = HybridJointImpedanceControl(
                joint_pos_current=self.arm.get_joint_positions(),
                Kq=self.arm.Kq_default,
                Kqd=self.arm.Kqd_default,
                Kx=self.arm.Kx_default,
                Kxd=self.arm.Kxd_default,
                robot_model=self.arm.robot_model,
                ignore_gravity=self.arm.use_grav_comp,
            )
        elif self.impedance_type == "joint":
            policy = JointImpedanceControl(
                joint_pos_current=self.arm.get_joint_positions(),
                Kp=self.arm.Kq_default,
                Kd=self.arm.Kqd_default,
                robot_model=self.arm.robot_model,
                ignore_gravity=self.arm.use_grav_comp,
            )
        else:
            raise ValueError(f"Unknown control type: {self.impedance_type}")

        # do not block until finished, since we want continuous control
        self.arm.send_torch_policy(policy, blocking=False)

        log.info("Reset complete. Rollout paused. Press SPACE to unpause.")
        self._get_user_input(paused=not self.human_control)

        obs = self._get_obs()
        if self.action_type == "joint":
            obs["target_joint_pos"] = obs["robot_state"][:7]
        elif self.action_type == "cartesian":
            obs["target_ee_pose"] = obs["ee_pose"]
        obs["target_gripper_pos"] = obs["robot_state"][-1:]

        info = self._get_info()

        return obs, info

    def close(self):
        # # TODO: close the connections to the robot once polymetis supports it
        # self.arm.close()
        # self.gripper.close()
        self.arm.terminate_current_policy()

        for camera in self.cameras.values():
            camera.close()

        # Clean up pygame
        pygame.quit()

    def _get_obs(self) -> ObsType:
        gripper_width = torch.tensor([self.gripper.get_state().width])

        state = self.arm.get_state_dict()

        robot_state = torch.cat([state["joint_pos"], gripper_width], dim=-1)

        ee_pose = state["ee_pose"]  # pos: (x,y,z) + quat: (x,y,z,w)

        # rearrange the quaternion to w,x,y,z convention
        ee_pos = ee_pose[:3]
        ee_wxyz = torch.cat((ee_pose[6:], ee_pose[3:6]), dim=-1)
        ee_pose = torch.cat((ee_pos, ee_wxyz), dim=0)

        ee_rot = quaternion_to_matrix(ee_wxyz)
        ee_transform = make_pose(ee_pos, ee_rot)

        obs_dict = {
            key: camera.get_observation() for key, camera in self.cameras.items()
        }

        # target values are added in step/reset methods
        obs_dict |= {
            "robot_state": robot_state,
            "ee_pose": ee_pose,
            "ee_transform": ee_transform,
        }

        # TODO: convert to TensorDict once SyncVectorEnv has been removed
        return obs_dict

    def _get_info(self) -> InfoType:
        return {}

    def _get_user_input(
        self, paused: bool = False
    ) -> None | Literal["success", "fail"]:
        """Check for keyboard input and handle reset/quit commands"""
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "fail"
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        log.info("Trajectory failed - will reset after current step")
                        return "fail"
                    elif event.key == pygame.K_RETURN:
                        log.info("Trajectory succeeded - will reset after current step")
                        return "success"
                    elif event.key == pygame.K_BACKSPACE:
                        raise DoneEvalSignal
                    elif event.key == pygame.K_SPACE:
                        if paused:
                            log.info("Rollout unpaused.")
                            return None

                        log.info("Rollout paused. Press SPACE to unpause.")
                        paused = True

            # no events to process
            if paused:
                time.sleep(0.1)
            else:
                return None
