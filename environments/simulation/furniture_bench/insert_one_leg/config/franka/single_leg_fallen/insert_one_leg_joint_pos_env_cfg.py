# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
from pathlib import Path

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
import isaaclab.sim as sim_utils

from ....furniture_bench_table_env_cfg import InsertOneLegEnvCfg
from ....mdp.events import (
    reset_table_parts_poses,
    randomize_object_position_from_predefined_area,
    randomize_light_intensity,
)
from ....mdp.terminations import success

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from ....assets.franka import FRANKA_PANDA_CFG

# Get current config directory
BASE_PATH = Path(__file__).parent.parent.parent.parent


@configclass
class EventCfg:
    """Configuration for events."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="startup",
        params={
            # "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400], # default initial pose from IsaacLab
            "default_pose": [
                0.1995,
                -0.2052,
                -0.2379,
                -2.5128,
                0.0027,
                2.3185,
                -0.6007,
                0.0400,
                0.0400,
            ],  # corresponds to the default initial posed which is also used for the real-world setup
        },
    )

    # set Franka gripper's dynamic and static frictions higher to simulate the black tape in the real world Franka's setup
    # instead of duplicating code, just reuse the existing function from IsaacLab (originally used for randomizing the friction properties) by setting the same upper and lower boundary values.
    # for more details: https://github.com/isaac-sim/IsaacLab/blob/1f0be3d2cc75c5019750d7873bb16845f9a4184c/source/isaaclab/isaaclab/envs/mdp/events.py#L148
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_.*finger"),
            "static_friction_range": (1.5, 1.5),
            "dynamic_friction_range": (1.5, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # change the robot's initial pose slightly
    randomize_franka_joint_state = EventTerm(
        func=franka_stack_events.randomize_joint_by_gaussian_offset,
        mode="reset",
        params={
            "mean": 0.0,
            "std": 0.02,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # reset the table-top part to the same pose on each reset
    reset_asset_poses = EventTerm(
        func=reset_table_parts_poses,
        mode="reset",
        params={
            "asset_cfgs": {
                "square_table_top": SceneEntityCfg("square_table_top"),
            },
            "initial_poses": {
                "square_table_top": {
                    "position": [0.515, 0.092, 0.0125],
                    "orientation": [0, 0, 0.7071068, 0.7071068],
                },
            },
        },
    )

    # randomize the table legs position
    randomize_object_position_from_predefined_area = EventTerm(
        func=randomize_object_position_from_predefined_area,
        mode="reset",
        params={
            "asset_cfgs": {
                "square_table_leg_1": SceneEntityCfg("square_table_leg_1"),
            },
            "reference_poses": {
                "square_table_leg_1": {
                    "position": [0.3, 0.0, 0.0125],
                    "orientation": [0.7071068, 0, 0, -0.7071068],
                },
            },
            "predefined_areas": [  # rectangular-area between the robot and the obstacle-assets
                {"x_min": 0.2, "x_max": 0.38, "y_min": -0.42, "y_max": 0.41},
                # free rectangular-area between the obstacle-assets and the square_table_top
                {"x_min": 0.50, "x_max": 0.55, "y_min": -0.14, "y_max": -0.02},
            ],
        },
    )

    randomize_light_intensity = EventTerm(
        func=randomize_light_intensity,
        mode="reset",
        params={
            "intensity_range": (20000, 50000),
            "asset_cfgs": [
                SceneEntityCfg("background_light"),
                SceneEntityCfg("front_light"),
            ],
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # success function used as a sparse reward function
    success = RewTerm(
        func=success,
        params={
            "xy_threshold": 0.0003,
            "height_threshold": 0.0002,
            "leg_target_frames": [
                "square_table_leg1_target_positions_frame",
            ],
        },
        weight=1.0,
    )


@configclass
class FrankaInsertOneLegEnvCfg(InsertOneLegEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set events
        self.events = EventCfg()

        # Set the reward function as a termination term as we currently do only imitation learning
        self.terminations.success = None
        self.terminations.success = DoneTerm(
            func=success,
            params={
                "xy_threshold": 0.0003,
                "height_threshold": 0.0002,
                "leg_target_frames": [
                    "square_table_leg1_target_positions_frame",
                ],
            },
            time_out=True,
        )

        # in this case, use the success function as a sparse reward
        self.rewards: RewardsCfg = RewardsCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]

        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={
                "panda_finger_.*": -0.02
            },  # further decreasing the value further leads to very aggressive controller reaction and penetrations with all objects because of very large forces
        )
        # Setup all static and dynamic objects, including their properties
        rigid_body_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=100,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
            max_contact_impulse=1.0,
            linear_damping=1.0,
            angular_damping=1.0,
        )

        collision_props_table_parts = sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.005,
            rest_offset=0.005,
        )

        # increasing the actual object's masses seems to imporve simulation stability.
        # Too high difference between the franka's arm mass and a manipulated object's mass is unstable.
        # E.g., between the object and the franka arm there are interpenetrations, also the object jitters during manipulation.
        obstacle_mass = sim_utils.MassPropertiesCfg(
            mass=0.065 * 10,
        )
        table_top_mass = sim_utils.MassPropertiesCfg(
            mass=0.167 * 10,
        )
        leg_mass = sim_utils.MassPropertiesCfg(
            mass=0.036 * 10,
        )

        # Obstacles
        static_body_properties = RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            solver_position_iteration_count=100,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
            max_contact_impulse=1.0,
            linear_damping=1.0,
            angular_damping=1.0,
        )
        #  front
        self.scene.obstacle_front = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObstacleFront",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.61, 0, 0.01], rot=[0.707, 0, 0, 0.707]
            ),
            spawn=UsdFileCfg(
                usd_path=os.path.join(BASE_PATH, "assets/obstacle_front.usd"),
                rigid_props=static_body_properties,
                mass_props=obstacle_mass,
                semantic_tags=[("class", "obstacle_front")],
            ),
        )
        # left side
        self.scene.obstacle_left_side = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObstacleLeftSide",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.535, 0.185, 0.01], rot=[0.707, 0, 0, 0.707]
            ),
            spawn=UsdFileCfg(
                usd_path=os.path.join(BASE_PATH, "assets/obstacle_side.usd"),
                rigid_props=static_body_properties,
                mass_props=obstacle_mass,
            ),
        )
        # right side
        self.scene.obstacle_right_side = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObstacleRightSide",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.535, -0.185, 0.01], rot=[0.707, 0, 0, 0.707]
            ),
            spawn=UsdFileCfg(
                usd_path=os.path.join(BASE_PATH, "assets/obstacle_side.usd"),
                rigid_props=static_body_properties,
                mass_props=obstacle_mass,
            ),
        )

        # square table parts
        self.scene.square_table_top = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SquareTable_Top",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.515, 0.092, 0.05], rot=[0, 0, 0.7071068, 0.7071068]
            ),
            spawn=UsdFileCfg(
                usd_path=os.path.join(BASE_PATH, "assets/square_table_top.usd"),
                rigid_props=rigid_body_properties,
                collision_props=collision_props_table_parts,
                mass_props=table_top_mass,
                semantic_tags=[("class", "square_table_top")],
            ),
        )

        self.scene.square_table_leg_1 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SquareTable_Leg_1",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.3, 0.0, 0.05], rot=[0.7071068, 0, 0, -0.7071068]
            ),
            spawn=UsdFileCfg(
                usd_path=os.path.join(BASE_PATH, "assets/square_table_leg1.usd"),
                rigid_props=rigid_body_properties,
                collision_props=collision_props_table_parts,
                mass_props=leg_mass,
                semantic_tags=[("class", "square_table_leg_1")],
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        # pos=[0.0, 0.0, 0.1034], # corresponds to the middle point of the original Franka gripper knobs
                        pos=[
                            0.0,
                            0.0,
                            0.209,
                        ],  # corresponds to the middle point of the UMI gripper knobs
                    ),
                ),
            ],
        )

        # reward relevant markers
        targets_marker_cfg = FRAME_MARKER_CFG.copy()
        targets_marker_cfg.markers["frame"].scale = (0.001, 0.001, 0.001)
        targets_marker_cfg.prim_path = "/Visuals/SquareTableTopTargetFrameTransformer"
        self.scene.square_table_top_target_positions_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/SquareTable_Top/square_table_top",  # make sure that the path is showing to the rigid-body prim which is not necessarily the root prim
            debug_vis=False,
            visualizer_cfg=targets_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/SquareTable_Top/square_table_top",
                    name="square_table_assembly_slot_1",
                    offset=OffsetCfg(
                        pos=(-0.05594, -0.00953, 0.056669),
                    ),
                ),
            ],
        )

        self.scene.square_table_leg1_target_positions_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/SquareTable_Leg_1/square_table_leg1",  # make sure that the path is showing to the rigid-body prim which is not necessarily the root prim
            debug_vis=True,
            visualizer_cfg=targets_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/SquareTable_Leg_1/square_table_leg1",
                    name="square_table_leg1_tip",
                    offset=OffsetCfg(
                        pos=(0.0, -0.0565, 0.0),
                    ),
                ),
            ],
        )
