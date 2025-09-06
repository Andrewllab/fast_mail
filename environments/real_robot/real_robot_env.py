import dataclasses
import logging
import time
from typing import Literal, Mapping

import gymnasium as gym
import pygame
import torch
from environments.real_robot.hardware.base_camera import BaseCamera
from environments.specs import ActionSpec, DataSpecs, ObsSpec, specs_to_spaces
from omegaconf import DictConfig
from polymetis import GripperInterface, RobotInterface
from torchcontrol.policies import CartesianImpedanceControl, HybridJointImpedanceControl
from utils.math import make_pose, normalize, quaternion_to_matrix

ObsType = dict[str, torch.Tensor | dict[str, torch.Tensor]]
InfoType = dict[str, torch.Tensor]

log = logging.getLogger(__name__)

GRIPPER_POS_SCALE = 0.04 / 0.07886763662099838


class RealRobotEnv(gym.Env):
    def __init__(
        self,
        robot: DictConfig,
        cameras: Mapping[str, BaseCamera] | None = None,
        control_type: Literal["cartesian", "hybrid_joint"] = "cartesian",
        binary_gripper_state: bool = True,
    ):
        self.control_type = control_type
        self.binary_gripper_state = binary_gripper_state

        # Initialize pygame for keyboard input
        pygame.init()
        self.screen = pygame.display.set_mode((100, 100))
        pygame.display.set_caption("Robot Control - Press K to reset, Q to quit")
        self.should_quit = False

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
            # we concatenate joint_pos and gripper_pos to get a shape of (T, 9) if binary_gripper_state is False
            "robot_state": (
                ObsSpec(elem_shape=(8,))
                if self.binary_gripper_state
                else ObsSpec(elem_shape=(9,))
            ),
            # xyz + wxyz quaternion
            "ee_pose": ObsSpec(elem_shape=(7,)),
            "target_ee_pose": ObsSpec(elem_shape=(7,)),
            # gripper_cam_transform
            "gripper_cam_transform": ObsSpec(elem_shape=(4, 4)),
        }

        camera_specs = {key: camera.spec for key, camera in self.cameras.items()}

        # add the dynamic_pose_obs_key to the gripper_cam if we have it
        # this is needed for the complete extrinsics of the moving gripper cam
        if "gripper_cam" in camera_specs:
            camera_specs["gripper_cam"] = dataclasses.replace(
                camera_specs["gripper_cam"],
                dynamic_pose_obs_key="gripper_cam_transform",
            )

        obs_specs.update(camera_specs)

        # absolute target pose (xyz + wxyz quaternion) + binary gripper command
        action = ActionSpec(action_dim=8)

        self._specs = DataSpecs(
            obs=obs_specs,
            action=action,
        )

        self.observation_space, self.action_space = specs_to_spaces(self._specs)

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def check_keyboard_input(self):
        """Check for keyboard input and handle reset/quit commands"""
        # Update the display to keep the window responsive
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_k:
                    log.info("Reset key pressed - will reset after current step")
                    return "reset"
                elif event.key == pygame.K_q:
                    log.info("Quit key pressed - setting quit flag")
                    self.should_quit = True
                    return "quit"
            elif event.type == pygame.QUIT:
                self.should_quit = True
                return "quit"
        return None

    def step(self, action: torch.Tensor) -> tuple[ObsType, float, bool, bool, InfoType]:
        # Check for keyboard input
        keyboard_action = self.check_keyboard_input()

        action = action.cpu()
        pos = action[:3]
        wxyz = action[3:7]
        wxyz = normalize(wxyz)
        gripper_command = action[7]
        xyzw = torch.cat((wxyz[-3:], wxyz[:-3]), dim=0)

        if self.control_type == "cartesian":
            self.arm.update_current_policy(
                {"ee_pos_desired": pos, "ee_quat_desired": xyzw}
            )
        elif self.control_type == "hybrid_joint":
            self.arm.update_desired_ee_pose(position=pos, orientation=xyzw)

        self.gripper.set_state(
            gripper_command.item(), speed=self.gripper_speed, force=self.gripper_force
        )

        obs = self._get_obs()
        obs["target_ee_pose"] = action[:7]  # xyz + wxyz quaternion
        info = self._get_info()

        # Add quit flag to info
        info["should_quit"] = self.should_quit

        # Handle reset request
        if keyboard_action == "reset":
            log.info("Resetting environment...")
            obs, reset_info = self.reset()
            info.update(reset_info)
            log.info("Waiting 5 seconds after reset...")
            time.sleep(5.0)
            log.info("Ready to continue")

        return obs, 0, False, False, info

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

        # wait a little longer in case the gripper is still movin
        time.sleep(1.0)

        # start the continuouos control policy
        if self.control_type == "cartesian":
            policy = CartesianImpedanceControl(
                joint_pos_current=self.arm.get_joint_positions(),
                Kp=self.arm.Kx_default,
                Kd=self.arm.Kxd_default,
                robot_model=self.arm.robot_model,
                ignore_gravity=self.arm.use_grav_comp,
            )
        elif self.control_type == "hybrid_joint":
            policy = HybridJointImpedanceControl(
                joint_pos_current=self.arm.get_joint_positions(),
                Kq=self.arm.Kq_default,
                Kqd=self.arm.Kqd_default,
                Kx=self.arm.Kx_default,
                Kxd=self.arm.Kxd_default,
                robot_model=self.arm.robot_model,
                ignore_gravity=self.arm.use_grav_comp,
            )
        else:
            raise ValueError(f"Unknown control type: {self.control_type}")

        # do not block until finished, since we want continuous control
        self.arm.send_torch_policy(policy, blocking=False)

        obs = self._get_obs()
        obs["target_ee_pose"] = obs["ee_pose"]
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
        # gripper in isaaclab and in the real world have different maximum widths
        gripper_width *= GRIPPER_POS_SCALE

        state = self.arm.get_robot_state()
        joint_pos = torch.tensor(state.joint_positions)

        if not self.binary_gripper_state:
            robot_state = torch.cat(
                (
                    joint_pos,  # 7
                    gripper_width,  # 1
                    -gripper_width,  # 1
                ),
                dim=-1,
            )
        else:
            thresh = (self.gripper_max_width * GRIPPER_POS_SCALE) / 2
            factor = 1.0 if gripper_width > thresh else -1.0
            robot_state = torch.cat(
                [
                    joint_pos,
                    factor
                    * torch.ones(1, dtype=joint_pos.dtype, device=joint_pos.device),
                ],
                dim=-1,
            )

        ee_pos, ee_xyzw = self.arm.robot_model.forward_kinematics(joint_pos)
        ee_wxyz = torch.cat((ee_xyzw[3:], ee_xyzw[:3]), dim=0)
        ee_rot = quaternion_to_matrix(ee_wxyz)
        ee_pose = torch.cat((ee_pos, ee_wxyz), dim=0)
        gripper_cam_transform = make_pose(ee_pos, ee_rot)

        obs_dict = {
            key: camera.get_observation() for key, camera in self.cameras.items()
        }

        obs_dict |= {
            "robot_state": robot_state,
            "ee_pose": ee_pose,
            # target_ee_pose is added in step/reset methods
            "gripper_cam_transform": gripper_cam_transform,
        }

        # TODO: convert to TensorDict once SyncVectorEnv has been removed
        return obs_dict

    def _get_info(self) -> InfoType:
        return {}


from functools import partial

from environments.wrappers import VectorToTorchWrapper
from gymnasium.vector import AutoresetMode, SyncVectorEnv


def make_env(**kwargs) -> gym.Env:

    env = partial(RealRobotEnv, **kwargs)

    # we need to disable automatic resets, since the agent predicts action
    # sequences
    env = SyncVectorEnv([env], copy=False, autoreset_mode=AutoresetMode.DISABLED)

    # gymnasium's VectorEnv converts Tensors to numpy arrays, so we need to
    # convert them back to Tensors
    env = VectorToTorchWrapper(env)

    # VecEnvs return a tuple of results whenever an attribute is accessed
    one_step_specs: DataSpecs = env.unwrapped.get_attr("specs")[0]

    # assign as new attribute so that GymEnvDataset can access it
    env.specs = one_step_specs

    return env
