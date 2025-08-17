import logging
import time

import hydra
import numpy as np
import rootutils
import torch
from omegaconf import DictConfig
from polymetis import RobotInterface

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from utils.math import convert_quat, euler_xyz_to_quaternion, quat_mul

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs", config_name="robot_shake_ee_pose"
)
def main(cfg: DictConfig) -> None:
    # connect to robot arm (we don't need the gripper)
    robot = RobotInterface(
        name=cfg.robot.name,
        ip_address=cfg.robot.ip_address,
        port=cfg.robot.arm_port,
        enforce_version=False,
    )
    log.info(f'Connected to robot "{cfg.robot.name}" at {cfg.robot.ip_address}')
    if (home_pose := cfg.robot.get("home_pose", None)) is not None:
        log.info(f"Setting home pose: {home_pose}")
        robot.set_home_pose(torch.tensor(home_pose))
    log.info(f"Going to home pose: {robot.home_pose}")
    robot.go_home()

    # a sequence of euler angles to go to
    waypoints_euler_xyz_deg = torch.tensor(cfg.waypoints_euler_xyz_deg)
    waypoints_euler_xyz = waypoints_euler_xyz_deg * (torch.pi / 180)

    if n_waypoints := len(waypoints_euler_xyz):
        # pass roll, pitch, yaw as position arguments
        waypoints_wxyz = euler_xyz_to_quaternion(*waypoints_euler_xyz.unbind(dim=-1))

        current_ee_pos, current_ee_xyzw = robot.get_ee_pose()
        current_ee_wxyz = convert_quat(current_ee_xyzw, to="wxyz")
        log.info(
            f"Current ee_pose={current_ee_pos.tolist() + current_ee_wxyz.tolist()}"
        )

        if cfg.get("relative"):
            # we left multiply the rotations here for some reason
            # polymetis also left multiplies rotation matrices if sending a delta ee_pose command
            waypoints_wxyz = quat_mul(
                waypoints_wxyz,
                # need to repeat the current orientation so that leading dims match
                current_ee_wxyz.repeat(n_waypoints, 1),
            )

        # convert to polymetis quaternion convention
        waypoints_xyzw = convert_quat(waypoints_wxyz, to="xyzw")

        # default (None) uses _adaptive_time_to_go
        time_to_go = cfg.get("time_to_go")
        wait_after = cfg.get("wait_after", 1.0)
        loop_waypoints = cfg.get("loop_waypoints", False)

        while True:
            for waypoint_xyzw in waypoints_xyzw:
                log.info(
                    f"Moving to waypoint: {current_ee_pos.tolist() + waypoint_xyzw.tolist()}"
                )
                time.sleep(wait_after)

                robot.move_to_ee_pose(
                    current_ee_pos.clone(),  # clone because the function modifies the input
                    waypoint_xyzw,
                    time_to_go=time_to_go,
                )

            if not loop_waypoints:
                break

    log.info("Waypoints exhausted.")


if __name__ == "__main__":
    main()
