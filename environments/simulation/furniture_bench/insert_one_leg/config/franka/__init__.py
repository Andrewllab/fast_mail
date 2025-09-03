# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os

import gymnasium as gym

from . import agents, four_legs_upright_all, single_leg_fallen

##
# Register Gym environments.
##


# A simplified version of the task Inser-One-Leg from Furniture-Bench: https://clvrai.github.io/furniture-bench/
# The simplification consists of initializing all four legs upright such that the agent can directly grasp them.

##
# Joint Position Control
##
gym.register(
    id="Isaac-Insert-One-Leg-UprightFour-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": four_legs_upright_all.insert_one_leg_joint_pos_env_cfg.FrankaInsertOneLegEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Insert-One-Leg-UprightFour-Franka-RGBD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": four_legs_upright_all.insert_one_leg_rgbd_env_cfg.FrankaInsertOneLegEnvCfg,
    },
    disable_env_checker=True,
)


##
# Inverse Kinematics - Relative Pose Control
##
gym.register(
    id="Isaac-Insert-One-Leg-Franka-UprightFour-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": four_legs_upright_all.insert_one_leg_ik_rel_env_cfg.FrankaInsertOneLegEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_low_dim.json"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Insert-One-Leg-Franka-RGBD-UprightFour-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": four_legs_upright_all.insert_one_leg_ik_rel_rgbd_env_cfg.FrankaInsertOneLegEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_low_dim.json"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Insert-One-Leg-Franka-UprightFour-Test_11_04_25_RGBD-IK-Rel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": four_legs_upright_all.insert_one_leg_test_11_04_25_ik_rel_rgbd_env_cfg.FrankaInsertOneLegEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_low_dim.json"
        ),
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##
gym.register(
    id="Isaac-Insert-One-Leg-Franka-UprightFour-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": four_legs_upright_all.insert_one_leg_ik_abs_env_cfg.FrankaInsertOneLegEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_low_dim.json"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Insert-One-Leg-Franka-RGBD-UprightFour-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": four_legs_upright_all.insert_one_leg_ik_abs_rgbd_env_cfg.FrankaInsertOneLegEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_low_dim.json"
        ),
    },
    disable_env_checker=True,
)

# A simplified version of the task Inser-One-Leg from Furniture-Bench: https://clvrai.github.io/furniture-bench/
# The simplification consists of initializing only one leg fallen, but randomly positioned each time.

##
# Joint Position Control
##

gym.register(
    id="Isaac-Insert-One-Leg-SingleFallen-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": single_leg_fallen.insert_one_leg_joint_pos_env_cfg.FrankaInsertOneLegEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Insert-One-Leg-SingleFallen-Franka-RGBD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": single_leg_fallen.insert_one_leg_rgbd_env_cfg.FrankaInsertOneLegEnvCfg,
    },
    disable_env_checker=True,
)


##
# Inverse Kinematics - Relative Pose Control
##

gym.register(
    id="Isaac-Insert-One-Leg-Franka-SingleFallen-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": single_leg_fallen.insert_one_leg_ik_rel_env_cfg.FrankaInsertOneLegEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_low_dim.json"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Insert-One-Leg-Franka-RGBD-SingleFallen-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": single_leg_fallen.insert_one_leg_ik_rel_rgbd_env_cfg.FrankaInsertOneLegEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_low_dim.json"
        ),
    },
    disable_env_checker=True,
)

##
# Inverse Kinematics - Absolute Pose Control
##

gym.register(
    id="Isaac-Insert-One-Leg-Franka-SingleFallen-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": single_leg_fallen.insert_one_leg_ik_abs_env_cfg.FrankaInsertOneLegEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_low_dim.json"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Insert-One-Leg-Franka-RGBD-SingleFallen-IK-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": single_leg_fallen.insert_one_leg_ik_abs_rgbd_env_cfg.FrankaInsertOneLegEnvCfg,
        "robomimic_bc_cfg_entry_point": os.path.join(
            agents.__path__[0], "robomimic/bc_rnn_low_dim.json"
        ),
    },
    disable_env_checker=True,
    # can't wrap the environment with any Gym wrappers, because the reset
    # method signature is incompatible with gym.Wrapper
    order_enforce=False,
)
