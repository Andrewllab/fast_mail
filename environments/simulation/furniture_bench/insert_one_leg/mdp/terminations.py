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
from tensordict import TensorDict

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def success(
    env: ManagerBasedRLEnv,
    leg_tip_pos_asset_cfgs: list[SceneEntityCfg],
    target_slot_pos_asset_cfg: SceneEntityCfg,
    xy_threshold: float,
    height_threshold: float,
) -> torch.Tensor:
    """Generalized reward function for the insert-one-leg furniture-bench task.
    The success is calculated based on the distance between the current leg's and pre-defined target 3d-positions in the world frame.

    Args:
        leg_tip_pos_asset_cfgs: list[SceneEntityCfg] for the tips of all legs (FrameTransformer).
        target_slot_pos_asset_cfg: SceneEntityCfg for the target slot frame (FrameTransformer).
        xy_threshold: threshold for the x- and y-axes under which the leg is considered as assembled.
        height_threshold: threshold for the z-axis under which the leg is considered as assembled.
        leg_target_frames: List of leg target frame names to check, e.g. in case there are multiple legs, check if at least one leg in assembled.
    Returns:
        torch.Tensor:
            True, if the euclidean distance between one of the legs and the target assembly slot is below both thresholds, otherwise False.
    """
    # extract 3d-positions in the world-frame
    square_table_top_assembly_slot_1_target_pos_w = env.scene[
        target_slot_pos_asset_cfg.name
    ].data.target_pos_w
    one_leg_inserted = False

    for leg_tip_pos_asset_cfg in leg_tip_pos_asset_cfgs:

        leg_tip_pos_w = env.scene[leg_tip_pos_asset_cfg.name].data.target_pos_w

        # calculate Euclidean distance
        position_dist_leg_assembly_slot = torch.linalg.vector_norm(
            square_table_top_assembly_slot_1_target_pos_w - leg_tip_pos_w, dim=1
        )

        # TODO: in case of vectorized-environments do not squeeze and check the following conditions for each environment instance separately.
        # Currently, the environment is used only for imitation learning where only one environment instance is enough.
        # TODO: Currently, the environment logic supports only one of the four assembly slots, but any leg is possible to be assembled

        # check if a leg is insterted
        xy_leg_inserted = torch.logical_and(
            position_dist_leg_assembly_slot[:, 0] <= xy_threshold,
            position_dist_leg_assembly_slot[:, 1] <= xy_threshold,
        )

        xyz_leg_inserted = torch.logical_and(
            xy_leg_inserted, position_dist_leg_assembly_slot[:, 2] <= height_threshold
        )

        # check if at least one leg is inserted
        one_leg_inserted |= xyz_leg_inserted

    return one_leg_inserted


def reached(
    env: ManagerBasedRLEnv,
    source_asset_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    threshold: float,
) -> bool:
    """Returns if a source assets's distance to a target position is lower that a pre-defined distance using L2-norm.
    Args:
        env: ManagerBasedRLEnv
        source_asset_cfg: SceneEntityCfg for the source entity (RigidObject or FrameTransformer).
        target_asset_cfg: SceneEntityCfg for the target entity (RigidObject or FrameTransformer).
        threshold: Distance threshold below which it is considered that the source has reached the target pose.
    Returns:
        torch.Tensor:
            True where the source is within threshold of the target, else False.
    """

    def extract_position(asset):
        match type(asset):
            case _ if isinstance(asset, FrameTransformer):
                return asset.data.target_pos_w

            case _ if isinstance(asset, RigidObject):
                return asset.data.body_pos_w

            case _:
                raise TypeError(f"Unsupported entity type: {type(asset)}")

    # extract asset objects from the scene
    source_asset = env.scene[source_asset_cfg.name]
    target_asset = env.scene[target_asset_cfg.name]

    # extract positions
    source_pos_w = extract_position(source_asset)
    target_pos_w = extract_position(target_asset)

    return torch.norm(target_pos_w - source_pos_w, dim=(1, 2)) <= threshold


def grasp_analysis(
    env: ManagerBasedRLEnv,
    contact_sensors_asset_cfgs: dict[str, SceneEntityCfg],
    ee_frame_asset_cfg: SceneEntityCfg,
    object_asset_cfg: SceneEntityCfg,
    reach_threshold: float,
    very_weak_grasp_threshold: float,
    weak_grasp_threshold: float,
    strong_grasp_threshold: float,
) -> TensorDict[str, torch.Tensor]:
    """Returns if a robot has successfully grasped an object, as well as the grasp type.

    The grasp is considered valid if:
      1) The end-effector has reached the target object within a pre-defined distance using L2-norm.
      2) Both fingers are in contact with the object and exceed either the very_weak, weak or strong grasp thresholds.

    # TODO: measure per-contact tangential forces and compute the wrenches to decide if a grasp is stable ot not. Currently, the decision is done purely based on the force [N] which is not optimal.

    Args:
        env: ManagerBasedRLEnv
        contact_sensors_asset_cfgs: Dict of SceneEntityCfg for each contact sensor
            (e.g., left and right fingers).
        ee_frame_asset_cfg: SceneEntityCfg for the end-effector frame (FrameTransformer).
        object_asset_cfg: SceneEntityCfg for the grasped object (RigidObject or FrameTransformer).
        reach_threshold: Distance [m] below which the end-effector is considered near the object.
        very_weak_grasp_threshold: Minimum normal force [N] for a very weak grasp.
        weak_grasp_threshold: Minimum normal force [N] for a weak grasp.
        strong_grasp_threshold: Minimum normal force [N] for a strong grasp.

    Returns:
        TensorDict[str, torch.Tensor]:
            grasp type and True where the object is considered grasped, else False, for each grasp.
    """

    # is the end-effector near the target object
    target_reached = reached(
        env, ee_frame_asset_cfg, object_asset_cfg, threshold=reach_threshold
    )

    # extract force vectors from all force sensors to compute the force per sensor
    force_vectors = {}
    for (
        contact_sensor_name,
        contact_sensor_asset_cfg,
    ) in contact_sensors_asset_cfgs.items():
        contact_sensor = env.scene[contact_sensor_asset_cfg.name]
        # TODO: Fix dimensions in case of vectorized environments
        # ContactSensor's force_matrix initialization has shape num_envs x history_len x num_bodies x num_filters x 3: https://github.com/isaac-sim/IsaacLab/blob/0f00ca2b4b2d54d5f90006a92abb1b00a72b2f20/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py#L336
        # according to data the first dim doesn't correspond to num_env: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor_data.py#L73
        force_vectors[contact_sensor_name] = torch.norm(
            contact_sensor.data.force_matrix_w
        )
    # use the minimum of both grasp forces as more robust metric
    min_force = torch.stack(list(force_vectors.values())).min()

    # compare forces against pre-defined thresholds to find the grasp type
    very_weak_grasp = torch.greater_equal(min_force, very_weak_grasp_threshold)
    very_weak_grasp = torch.logical_and(
        very_weak_grasp, torch.less(min_force, weak_grasp_threshold)
    )
    very_weak_grasp = torch.logical_and(very_weak_grasp, target_reached)
    weak_grasp = torch.greater_equal(min_force, weak_grasp_threshold)
    weak_grasp = torch.logical_and(
        weak_grasp, torch.less(min_force, strong_grasp_threshold)
    )
    weak_grasp = torch.logical_and(weak_grasp, target_reached)
    strong_grasp = torch.greater_equal(min_force, strong_grasp_threshold)
    strong_grasp = torch.logical_and(strong_grasp, target_reached)

    return TensorDict(
        {
            "very_weak_grasp": very_weak_grasp,
            "weak_grasp": weak_grasp,
            "strong_grasp": strong_grasp,
        }
    )


def lifted(
    env: ManagerBasedRLEnv,
    contact_sensors_asset_cfgs: dict[str, SceneEntityCfg],
    ee_frame_asset_cfg: SceneEntityCfg,
    object_asset_cfg: SceneEntityCfg,
    reach_threshold: float,
    very_weak_grasp_threshold: float,
    weak_grasp_threshold: float,
    strong_grasp_threshold: float,
    lift_threshold: float,
) -> torch.Tensor:
    """Returns if an object has been lifted successfully.

    An object is considered lifted if:
      1) The end-effector has reached it (within reach_threshold).
      2) The object is grasped (very weak, weak or strong grasp).
      3) The object's height in world frame exceeds lift_threshold.

    Args:
        env: ManagerBasedRLEnv
        contact_sensors_asset_cfgs: Dict of SceneEntityCfg for each contact sensor
            (e.g., left and right fingers).
        ee_frame_asset_cfg: SceneEntityCfg for the end-effector frame.
        object_asset_cfg: SceneEntityCfg for the object to lift.
        reach_threshold: Distance [m] below which the end-effector is considered near the object.
        very_weak_grasp_threshold: Minimum normal force [N] for a very weak grasp.
        weak_grasp_threshold: Minimum normal force [N] for a weak grasp.
        strong_grasp_threshold: Minimum normal force [N] for a strong grasp.
        lift_threshold: Minimum object height [m] above the world frame origin.

    Returns:
        torch.Tensor:
            True where the object is considered lifted, else False.
    """
    # check if the end-effector is near the object asset
    object_asset = env.scene[object_asset_cfg.name]
    object_asset_pos_w = object_asset.data.body_pos_w.clone()
    # TODO: remove when vectorization is everywhere supported
    object_asset_pos_w = object_asset_pos_w.squeeze(0)

    # analyze if reached and grasped
    grasped_analysis = grasp_analysis(
        env,
        contact_sensors_asset_cfgs,
        ee_frame_asset_cfg,
        object_asset_cfg,
        reach_threshold,
        very_weak_grasp_threshold,
        weak_grasp_threshold,
        strong_grasp_threshold,
    )

    # TODO: Remove quick workaround of the fact that tensordict doesn't support .any().
    # extract if at least one of the grasp types fulfilled
    object_grasped = torch.any(torch.stack(list(grasped_analysis.values())))
    object_lifted = torch.greater(object_asset_pos_w[:, 2], lift_threshold)
    object_lifted = torch.logical_and(object_grasped, object_lifted)

    return object_lifted


def inserted(
    env: ManagerBasedRLEnv,
    object_tip_pos_asset_cfg: SceneEntityCfg,
    target_pos_asset_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Returns if an object has been inserted into a target slot.

    The function computes the Euclidean distance between the target slot and the object (in world frame) and computes whether it is below a threshold.

    Args:
        env: ManagerBasedRLEnv.
        object_tip_pos_asset_cfg: SceneEntityCfg for the object frame (FrameTransformer).
        target_pos_asset_cfg: SceneEntityCfg for the target slot frame (FrameTransformer).
        threshold: Euclidean-Distance [m] below which insertion is considered successful.

    Returns:
        torch.Tensor:
            True where the object is considered inserted, else False.
    """

    object_tip_pos_asset: FrameTransformer = env.scene[object_tip_pos_asset_cfg.name]
    object_tip_pos_asset_pos_w = object_tip_pos_asset.data.target_pos_w
    square_table_target_pos_asset: FrameTransformer = env.scene[
        target_pos_asset_cfg.name
    ]
    square_table_target_pos_w = square_table_target_pos_asset.data.target_pos_w

    object_inserted = torch.less_equal(
        torch.norm(square_table_target_pos_w - object_tip_pos_asset_pos_w, dim=(1, 2)),
        threshold,
    )

    return object_inserted


def z_axis_aligned(
    env: ManagerBasedRLEnv,
    object_tip_pos_asset_cfg: SceneEntityCfg,
    target_pos_asset_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Axis-wise orientation alignment check between a target and object's poses.

    The function computes the relative rotation R_rel that rotates the object frame onto the target frame,
    then measure per-axis misalignment: angles_x/y/z = arccos(diag(R_rel)).

    Args:
        env: ManagerBasedRLEnv.
        object_tip_pos_asset_cfg: SceneEntityCfg object frame providing current/source orientation (FrameTransformer).
        target_pos_asset_cfg: SceneEntityCfg providing target orientation (FrameTransformer).
        threshold: Max allowed misalignment (rad, where rad x 180/π = degree).

    Returns:
        torch.Tensor:
            True where the z-axis misalignment is <= threshold, else False.
    """
    # TODO: remove squeeze when vectorized environments are supported again
    # extract poses from target and object assets
    square_table_target_position_asset: FrameTransformer = env.scene[
        target_pos_asset_cfg.name
    ]
    square_table_target_pos_w = (
        square_table_target_position_asset.data.target_pos_w.clone().squeeze(0)
    )
    square_table_target_quat_w = (
        square_table_target_position_asset.data.target_quat_w.clone().squeeze(0)
    )
    object_tip_pos_asset: FrameTransformer = env.scene[object_tip_pos_asset_cfg.name]
    object_tip_pos_w = object_tip_pos_asset.data.target_pos_w.clone().squeeze(0)
    object_tip_quat_w = object_tip_pos_asset.data.target_quat_w.clone().squeeze(0)

    # A rotation between two objects expresses how each axis of the source(object) frame aligns with each axis of the desired(target) frame (R_err = R_object.T * R_target​).
    # The diagonal entries of rot_matrix_rel_error are dot products of corresponding axes, where each axis is a basis vector (matrix column):
    # References:
    # 1) Chapter 2.2, Page 40, Equation 2.3: https://people.disim.univaq.it/~costanzo.manes/EDU_stuff/Robotics_Modelling,%20Planning%20and%20Control_Sciavicco_extract.pdf
    # 2) Chapter 2, Page 22, Equation 2.3: https://marsuniversity.github.io/ece387/Introduction-to-Robotics-Craig.pdf
    # calculate the relative rotation as quaternion (equivalent to computing it as R_err = R_obj.T * R_target).
    pos_rel_error, quat_rel_error = math_utils.compute_pose_error(
        object_tip_pos_w,
        object_tip_quat_w,
        square_table_target_pos_w,
        square_table_target_quat_w,
        rot_error_type="quat",
    )
    # Represent relative rotation between both objects in matrix form.
    rot_matrix_rel_error = math_utils.matrix_from_quat(quat_rel_error)
    # cosine of the alignment angles for each axis (x, y, z)
    axis_misalignment_cos_angles = rot_matrix_rel_error.diagonal(dim1=1, dim2=2)
    # The dot product of two unit vectors is equal to the cos(angle): https://en.wikipedia.org/wiki/Dot_product.
    # To calculate the per-axis angle compute the arccos of each diagonal element.
    # Misalignment angles in radians (per axis)
    axis_misalignment_angles = torch.arccos(axis_misalignment_cos_angles)

    return torch.less_equal(axis_misalignment_angles[:, -1], threshold)
