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
):
    """Second version of a reward function for the insert-one-leg furniture bench task. 
    The success is calculated based on the distance between the current leg's 3d-position and pre-defined target 3d-position in the world frame.

    Args:
        xy_threshold: threshold for the x- and y-axes under which the leg is considered as assembled.

        height_threshold: threshold for the z-axis under which the leg is considered as assembled.

    Returns:
        True, if the euclidean distance between one of the legs and the target assembly point is below the threshold, otherwise False.  
    """
    # extract 3d-positions in the world-frame 
    square_table_top_assembly_slot_1_target_pos = env.scene["square_table_top_target_positions_frame"].data.target_pos_w
    square_table_top_leg1_tip = env.scene["square_table_leg1_target_positions_frame"].data.target_pos_w
    square_table_top_leg2_tip = env.scene["square_table_leg2_target_positions_frame"].data.target_pos_w
    square_table_top_leg3_tip = env.scene["square_table_leg3_target_positions_frame"].data.target_pos_w
    square_table_top_leg4_tip = env.scene["square_table_leg4_target_positions_frame"].data.target_pos_w

    # calculate Euclidean distance
    position_dist_leg_1_assembly_slot_1 = torch.linalg.vector_norm(square_table_top_assembly_slot_1_target_pos - square_table_top_leg1_tip, dim=1).squeeze(0) 
    position_dist_leg_2_assembly_slot_1 = torch.linalg.vector_norm(square_table_top_assembly_slot_1_target_pos - square_table_top_leg2_tip, dim=1).squeeze(0) 
    position_dist_leg_3_assembly_slot_1 = torch.linalg.vector_norm(square_table_top_assembly_slot_1_target_pos - square_table_top_leg3_tip, dim=1).squeeze(0) 
    position_dist_leg_4_assembly_slot_1 = torch.linalg.vector_norm(square_table_top_assembly_slot_1_target_pos - square_table_top_leg4_tip, dim=1).squeeze(0) 

    # TODO: in case of vectorized-environments do not squeeze and check the following conditions for each environment instance separately. 
    # Currently, the environment is used only for imitation learning where only one environment instance is enough.
    # TODO: Currently, the environment logic supports only one of the four assembly slots, but any leg is possible to be assembled

    # check if leg_1 is insterted
    xy_leg1_inserted = torch.logical_and(position_dist_leg_1_assembly_slot_1[0] <= xy_threshold, 
                                        position_dist_leg_1_assembly_slot_1[1] <= xy_threshold)
    xyz_leg1_inserted = torch.logical_and(xy_leg1_inserted, 
                                    position_dist_leg_1_assembly_slot_1[2] <= height_threshold)

    # check if leg_2 is insterted
    xy_leg2_inserted = torch.logical_and(position_dist_leg_2_assembly_slot_1[0] <= xy_threshold, 
                                        position_dist_leg_2_assembly_slot_1[1] <= xy_threshold)
    xyz_leg2_inserted = torch.logical_and(xy_leg2_inserted, 
                                    position_dist_leg_2_assembly_slot_1[2] <= height_threshold)

    # check if leg_3 is insterted
    xy_leg3_inserted = torch.logical_and(position_dist_leg_3_assembly_slot_1[0] <= xy_threshold, 
                                        position_dist_leg_3_assembly_slot_1[1] <= xy_threshold)
    xyz_leg3_inserted = torch.logical_and(xy_leg3_inserted, 
                                    position_dist_leg_3_assembly_slot_1[2] <= height_threshold)

    # check if leg_4 is insterted
    xy_leg4_inserted = torch.logical_and(position_dist_leg_4_assembly_slot_1[0] <= xy_threshold, 
                                        position_dist_leg_4_assembly_slot_1[1] <= xy_threshold)
    xyz_leg4_inserted = torch.logical_and(xy_leg4_inserted, 
                                    position_dist_leg_4_assembly_slot_1[2] <= height_threshold)

    # check if at least one leg is inserted
    one_leg_inserted = xyz_leg1_inserted | xyz_leg2_inserted | xyz_leg3_inserted | xyz_leg4_inserted
    one_leg_inserted = one_leg_inserted.unsqueeze(0)

    return one_leg_inserted
