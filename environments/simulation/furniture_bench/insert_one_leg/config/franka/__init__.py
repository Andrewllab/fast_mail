# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import gymnasium as gym
import os

from . import (
    agents,
    insert_one_leg_ik_rel_env_cfg,
    insert_one_leg_ik_rel_rgbd_env_cfg,
    insert_one_leg_rgbd_env_cfg,
    insert_one_leg_joint_pos_env_cfg,
)

##
# Register Gym environments.
##

##
# Joint Position Control
##

gym.register(
    id="Isaac-Insert-One-Leg-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": insert_one_leg_joint_pos_env_cfg.FrankaInsertOneLegEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Insert-One-Leg-Franka-RGBD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={    
        "env_cfg_entry_point": insert_one_leg_rgbd_env_cfg.FrankaInsertOneLegEnvCfg,
    },
    disable_env_checker=True,
)


##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Insert-One-Leg-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": insert_one_leg_ik_rel_env_cfg.FrankaInsertOneLegEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Insert-One-Leg-Franka-RGBD-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": insert_one_leg_ik_rel_rgbd_env_cfg.FrankaInsertOneLegEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(agents.__path__[0], "robomimic/bc_rnn_low_dim.json"),
    },
    disable_env_checker=True,
)
