# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def success(
    env: ManagerBasedRLEnv,
    xy_threshold: float,
    height_threshold: float,
    leg_target_frames: list[str],
):
    """Generalized reward function for the insert-one-leg furniture-bench task. 
    The success is calculated based on the distance between the current leg's and pre-defined target 3d-positions in the world frame.

    Args:
        xy_threshold: threshold for the x- and y-axes under which the leg is considered as assembled.
        height_threshold: threshold for the z-axis under which the leg is considered as assembled.
        leg_target_frames: List of leg target frame names to check.
    Returns:
        True, if the euclidean distance between one of the legs and the target assembly slot is below both thresholds, otherwise False.  
    """
    # extract 3d-positions in the world-frame 
    square_table_top_assembly_slot_1_target_pos = env.scene["square_table_top_target_positions_frame"].data.target_pos_w

    one_leg_inserted = False

    for leg_target_frame in leg_target_frames:

        square_table_leg_tip_pos = env.scene[leg_target_frame].data.target_pos_w

        # calculate Euclidean distance
        position_dist_leg_assembly_slot = torch.linalg.vector_norm(square_table_top_assembly_slot_1_target_pos - square_table_leg_tip_pos, dim=1).squeeze(0) 

        # TODO: in case of vectorized-environments do not squeeze and check the following conditions for each environment instance separately. 
        # Currently, the environment is used only for imitation learning where only one environment instance is enough.
        # TODO: Currently, the environment logic supports only one of the four assembly slots, but any leg is possible to be assembled

        # check if a leg is insterted
        xy_leg_inserted = torch.logical_and(position_dist_leg_assembly_slot[0] <= xy_threshold, 
                                            position_dist_leg_assembly_slot[1] <= xy_threshold)
        xyz_leg_inserted = torch.logical_and(xy_leg_inserted, 
                                        position_dist_leg_assembly_slot[2] <= height_threshold)

        # check if at least one leg is inserted
        one_leg_inserted |= xyz_leg_inserted

    return one_leg_inserted
