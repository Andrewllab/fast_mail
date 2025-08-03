# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""
from typing import Union

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Keyboard teleoperation for IsaacLab environments."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Insert-One-Leg-Franka-SingleFallen-IK-Rel-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Device for interacting with environment.",
)
parser.add_argument(
    "--sensitivity", type=float, default=0.5, help="Sensitivity factor."
)
parser.add_argument(
    "--simpub",
    action="store_true",
    default=False,
    help="Enable SimPub.",
)
parser.add_argument("--step_hz", type=int, help="Environment stepping rate in Hz.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
app_launcher_args = vars(args_cli)

if args_cli.teleop_device.lower() == "handtracking":
    app_launcher_args["experience"] = (
        f"{os.environ['ISAACLAB_PATH']}/apps/isaaclab.python.xr.openxr.kit"
    )

if "simpub" in args_cli.teleop_device.lower():
    args_cli.simpub = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import gymnasium as gym
import omni.log  # noqa: F401
import torch

from isaaclab.devices import Se3Gamepad, Se3Keyboard, Se3SpaceMouse
from isaaclab.envs import ViewerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab_tasks.utils import parse_env_cfg

# TODO: remove the nasty hack of solving the ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import furniture_bench


# from evdev import InputDevice, list_devices
# import hid  # this is the hidapi backend used by IsaacLab
# # required to find the spacemouse device
# def patched_find_device(self):
#     devices = [InputDevice(path) for path in list_devices()]
#     for device in devices:
#         if any(name in device.name for name in ["Space", "Navigator", "3Dconnexion"]):
#             print(f"[Patched] Found SpaceMouse-like device: {device.name} @ {device.path}")
#             self._input_device = device

#             # Now find the HID device matching vendor and product ID
#             for d in hid.enumerate():
#                 if (d["vendor_id"] == 0x046d and d["product_id"] in [0xc626, 0xc62b]):  # Logitech / 3Dconnexion
#                     self._device = hid.device()
#                     self._device.open_path(d["path"])
#                     print(f"[Patched] HID device opened: {d['product_string']}")
#                     return

#     raise OSError("No compatible HID device found for SpaceMouse. (patched version)")
# # Patch the method
# Se3SpaceMouse._find_device = patched_find_device


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def pre_process_actions(
    teleop_output: Union[dict, tuple], device: str, num_envs: int
) -> torch.Tensor:
    """Pre-process actions for the environment."""

    match teleop_output:
        case dict():
            raw_action = teleop_output["action"]
            gripper_command = teleop_output["gripper"]
            action = raw_action.repeat(num_envs, 1)
        case tuple():
            raw_action, gripper_command = teleop_output
            # convert to torch
            action = torch.tensor(raw_action, dtype=torch.float, device=device).repeat(
                num_envs, 1
            )
        case _:
            raise ValueError(
                f"Unsupported teleoperation output type'{type(teleop_output)}'. Supported: 'dict', 'tuple'."
            )

    # resolve gripper command
    gripper_vel = torch.zeros(action.shape[0], 1, device=action.device)
    gripper_vel[:] = -1.0 if gripper_command else 1.0
    # compute actions
    return torch.concat([action, gripper_vel], dim=1)


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""

    # if handtracking is selected, rate limiting is achieved via OpenXR
    rate_limiter = (
        None
        if args_cli.teleop_device.lower() == "handtracking" or args_cli.step_hz == 0
        else RateLimiter(args_cli.step_hz)
    )

    # parse configuration
    env_cfg = parse_env_cfg(
        task_name=args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # remove timeout
    # WARNING: this attribute does not exist for non-manager-based envs
    env_cfg.terminations.time_out = None

    # reset teleop if env is reset
    def teleop_reset(env, env_ids):
        teleop_interface.reset()

    env_cfg.events.reset_teleop = EventTerm(func=teleop_reset, mode="reset")

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    print(
        f"Teleop-interface action: Action: {env.action_manager._action, env.action_manager._prev_action, env.action_space, env.action_manager.action_term_dim}"
    )

    # TODO: do we still need to call this???
    env.setup_manager_visualizers()
    # add teleoperation key for env reset
    should_reset_recording_instance = False

    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True

    # enable simpub if selected
    if (
        args_cli.simpub
        and env.unwrapped.sim is not None
        and env.unwrapped.sim.stage is not None
    ):
        print("parsing usd stage...")
        # WARNING: Change to your PC's IP!!!
        IsaacSimPublisher(host="192.168.0.110", stage=env.unwrapped.sim.stage)

    match args_cli.teleop_device.lower():

        case "keyboard":
            teleop_interface = Se3Keyboard(
                pos_sensitivity=0.1 * args_cli.sensitivity,
                rot_sensitivity=0.8 * args_cli.sensitivity,
            )
            teleop_interface.add_callback("R", reset_recording_instance)

        case "spacemouse":
            teleop_interface = Se3SpaceMouse(
                pos_sensitivity=0.1 * args_cli.sensitivity,
                rot_sensitivity=0.8 * args_cli.sensitivity,
            )

        case "gamepad":
            from carb.input import GamepadInput

            teleop_interface = Se3Gamepad(
                pos_sensitivity=0.1 * args_cli.sensitivity,
                rot_sensitivity=0.1 * args_cli.sensitivity,
                dead_zone=0.2,
            )
            teleop_interface.add_callback(GamepadInput.B, reset_recording_instance)

        case "handtracking":
            from isaacsim.xr.openxr import OpenXRSpec

            teleop_interface = Se3HandTracking(
                OpenXRSpec.XrHandEXT.XR_HAND_RIGHT_EXT, False, True
            )
            teleop_interface.add_callback("RESET", reset_recording_instance)
            viewer = ViewerCfg(
                eye=(-0.25, -0.3, 0.5), lookat=(0.6, 0, 0), asset_name="viewer"
            )
            ViewportCameraController(env, viewer)

        case "simpub":
            from simpub.sim.isaacsim_publisher import IsaacSimPublisher

            # infer the mode automatically depending on the task name
            controller_mode = (
                "relative" if "rel" in args_cli.task.lower() else "absolute"
            )
            teleop_interface = Se3IKSimPubHandTracking(
                hand="right",
                delta_pos_scale_factor=1.0,
                delta_rot_scale_factor=1.0,
                controller_mode=controller_mode,
                eef_pos_offset=torch.tensor(
                    [[0.0, 0.0, 0.2034]]
                ),  # NOTE: add a small positional offset along the z-axis, avoiding the overlap between the motion controller required to be exactly at the eef's pose.
                device_name="ALRMetaQuest3",  # NOTE: change if you use a different MetaQuest3 device
            )
            env.reset()
            # use the current eef pose from simulation
            #   ik-abs: as the target eef's pose at the very beginning where there is no data comming from the motion controller
            #   ik-rel: as debugging data
            cur_eef_pos = (
                env.scene["ee_frame"].data.target_pos_w.clone().cpu().squeeze(0)
            )
            cur_eef_rot_quat_w = (
                env.scene["ee_frame"].data.target_quat_w.clone().cpu().squeeze(0)
            )  # (w, x, y, z) https://github.com/isaac-sim/IsaacLab/blob/f22b5eb80172199fc4c25d34e01eb09acdff08b6/source/isaaclab/isaaclab/sensors/frame_transformer/frame_transformer_data.py#L40C11-L41C15
            teleop_interface.update_eef_pose(
                cur_eef_pos=cur_eef_pos, cur_eef_rot_quat_w=cur_eef_rot_quat_w
            )

        case _:
            raise ValueError(
                f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse', 'handtracking', 'simpub'."
            )

    print(teleop_interface)

    # reset environment
    env.reset()

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            if args_cli.teleop_device.lower() == "simpub":
                # use the current eef pose from simulation
                #   ik-abs: as the target eef's pose at the very beginning where there is no data comming from the motion controller
                #   ik-rel: as debugging data
                cur_eef_pos = (
                    env.scene["ee_frame"].data.target_pos_w.clone().cpu().squeeze(0)
                )
                cur_eef_rot_quat_w = (
                    env.scene["ee_frame"].data.target_quat_w.clone().cpu().squeeze(0)
                )  # (w, x, y, z) https://github.com/isaac-sim/IsaacLab/blob/f22b5eb80172199fc4c25d34e01eb09acdff08b6/source/isaaclab/isaaclab/sensors/frame_transformer/frame_transformer_data.py#L40C11-L41C15
                teleop_interface.update_eef_pose(
                    cur_eef_pos=cur_eef_pos, cur_eef_rot_quat_w=cur_eef_rot_quat_w
                )
            # get teleop-device command
            teleop_output = teleop_interface.advance()
            # pre-process actions
            actions = pre_process_actions(
                teleop_output, device=env.device, num_envs=env.num_envs
            )

            # apply actions
            env.step(actions)

            if should_reset_recording_instance:
                env.reset()
                should_reset_recording_instance = False

            # check that simulation is stopped or not
            if env.sim.is_stopped():
                break

            if rate_limiter:
                rate_limiter.sleep(env)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
