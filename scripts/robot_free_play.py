import logging

import hydra
import numpy as np
import pygame
import rootutils
import torch
from omegaconf import DictConfig
from polymetis import RobotInterface

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from environments.real_robot.hardware.franka_control import HumanControl
from utils.math import convert_quat

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="robot_free_play")
def main(cfg: DictConfig) -> None:
    # connect to robot arm (we don't need the gripper)
    robot = RobotInterface(
        name=cfg.robot.name,
        ip_address=cfg.robot.ip_address,
        port=cfg.robot.arm_port,
        enforce_version=False,
    )
    log.info(f'Connected to robot "{cfg.robot.name}" at {cfg.robot.ip_address}')
    if cfg.get("go_home", False):
        if (home_pose := cfg.robot.get("home_pose", None)) is not None:
            log.info(f"Setting home pose: {home_pose}")
            robot.set_home_pose(torch.tensor(home_pose))

        robot.go_home()

    robot.send_torch_policy(HumanControl(robot), blocking=False)

    # default (None) leaves it as 4
    torch.set_printoptions(
        precision=cfg.get("print_precision"), linewidth=cfg.get("print_linewidth")
    )

    fps = cfg.get("fps", 5)
    clock = pygame.time.Clock()

    pygame.init()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        current_joint_pos = robot.get_joint_positions()
        current_ee_pos, current_ee_xyzw = robot.robot_model.forward_kinematics(
            current_joint_pos
        )
        current_ee_wxyz = convert_quat(current_ee_xyzw, to="wxyz")
        log.info(
            "Current state:\n"
            f"Joint positions: {current_joint_pos}\n"
            f"ee position: {current_ee_pos}\n"
            f"ee orientation (wxyz): {current_ee_wxyz}\n"
        )

        clock.tick(fps)


if __name__ == "__main__":
    main()
