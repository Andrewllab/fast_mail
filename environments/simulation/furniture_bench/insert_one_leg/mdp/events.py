# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
import random
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBase
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)


def randomize_light_intensity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    asset_cfgs: list[SceneEntityCfg],
):
    # TODO: Currently, it does not support vertorized envs.
    for asset_cfg in asset_cfgs:
        asset: AssetBase = env.scene[asset_cfg.name]
        light_prim = asset.prims[0]

        # Sample new light intensity
        new_intensity = random.uniform(intensity_range[0], intensity_range[1])

        # Set light intensity to light prim
        intensity_attr = light_prim.GetAttribute("inputs:intensity")
        intensity_attr.Set(new_intensity)


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pose_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_list = []

    for i in range(num_objects):
        for j in range(max_sample_tries):
            sample = [random.uniform(range[0], range[1]) for range in range_list]

            # Accept pose if it is the first one, or if reached max num tries
            if len(pose_list) == 0 or j == max_sample_tries - 1:
                pose_list.append(sample)
                break

            # Check if pose of object is sufficiently far away from all other objects
            separation_check = [math.dist(sample[:3], pose[:3]) > min_separation for pose in pose_list]
            if False not in separation_check:
                pose_list.append(sample)
                break

    return pose_list


def randomize_object_position(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: dict[str, SceneEntityCfg],
    reference_poses: dict[str, dict], 
    variation: float,
    min_separation: float = 0.1,
    max_sample_tries: int = 5000,
):
    if env_ids is None:
        return

    new_xy_positions = {}
    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        for asset_name, asset_pose in reference_poses.items(): 
            for _ in range (max_sample_tries):
                ref_x, ref_y, ref_z = asset_pose["position"][:3]
                new_x = random.uniform(ref_x - variation, ref_x + variation)
                new_y = random.uniform(ref_y - variation, ref_y + variation)

                # check for collisitions between new object positions
                if all(math.dist([new_x, new_y], [px, py]) > min_separation for px, py, _ in new_xy_positions.values()):
                    new_xy_positions[asset_name] = torch.tensor([new_x, new_y, ref_z], device=env.device)
                    break

        # set the chosen randomized asset xy-positions
        for asset_name, asset_cfg in asset_cfgs.items():
            asset = env.scene[asset_cfg.name]

            # take original orientation
            orientation = torch.tensor(reference_poses[asset_name]["orientation"], device=env.device)
            # Write pose to simulation
            asset.write_root_pose_to_sim(
                torch.cat([new_xy_positions[asset_name], orientation], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )


def reset_table_parts_poses(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfgs: dict[str, SceneEntityCfg],
    initial_poses: dict[str, SceneEntityCfg],
):
    if env_ids is None:
        return

    # Set poses in each environment independently
    for cur_env in env_ids.tolist():
        # Extract asset and its initial pose  
        for name, asset_cfg in asset_cfgs.items():
            asset = env.scene[asset_cfg.name]

            # Write asset pose to simulation
            position_tensor = torch.tensor([initial_poses[name]["position"]], device=env.device)
            orientation_tensor = torch.tensor([initial_poses[name]["orientation"]], device=env.device)
            asset.write_root_pose_to_sim(
                torch.cat([position_tensor, orientation_tensor], dim=-1), env_ids=torch.tensor([cur_env], device=env.device)
            )
            # set velocity to zero, otherwise they get carried from the previous episode 
            asset.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=env.device), env_ids=torch.tensor([cur_env], device=env.device)
            )
