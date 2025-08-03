# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import insert_one_leg_rgbd_env_cfg

##
# Pre-defined configs
##
from ....assets.franka import FRANKA_ORIGINAL_PANDA_UMI_GRIPPERS_HIGH_PD_CFG


@configclass
class FrankaInsertOneLegEnvCfg(insert_one_leg_rgbd_env_cfg.FrankaInsertOneLegEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_ORIGINAL_PANDA_UMI_GRIPPERS_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # Set actions for the specific robot type (franka)
        # Disable gravity for easier control
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.209]
            ),  # correponds to the tip point of the UMI grippers knobs
            # body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]), # correponds to the tip point of the original grippers knobs
        )
