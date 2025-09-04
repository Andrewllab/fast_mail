# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
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
    set_joint_pose,
)
from ....mdp.terminations import (
    success,
    reached,
    grasp_analysis,
    lifted,
    inserted,
    z_axis_aligned,
)
from .....insert_one_leg import assets

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from ....assets.franka import FRANKA_PANDA_CFG


@configclass
class EvalMetricsObsGroupCfg(ObsGroup):
    """An observation group acting as a container for all relevant evaluation metrics."""

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


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

    # set_franka_arm_pose = EventTerm(
    #     func=set_joint_pose,
    #     mode="reset",
    #     params={
    #         # "default_pose": [0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400], # default initial pose from IsaacLab
    #         "desired_pose": [0.005649, -0.099542, -0.118758, -2.201725, -0.009681, 2.159746, -0.908100, 0.0400, 0.0400] # corresponds to the initial pose on the real-world setup
    #     },
    # )

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
            "leg_tip_pos_asset_cfgs": [
                SceneEntityCfg("square_table_leg1_target_positions_frame"),
            ],
            "target_slot_pos_asset_cfg": SceneEntityCfg(
                "square_table_top_target_positions_frame"
            ),
            "xy_threshold": 0.0003,
            "height_threshold": 0.0002,
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
                "leg_tip_pos_asset_cfgs": [
                    SceneEntityCfg("square_table_leg1_target_positions_frame"),
                ],
                "target_slot_pos_asset_cfg": SceneEntityCfg(
                    "square_table_top_target_positions_frame"
                ),
                "xy_threshold": 0.0003,
                "height_threshold": 0.0002,
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

        # Add all evaluation metrics as one observation group. If desired, they can also be used as terminations/rewards/events as well.
        self.observations.eval_metrics = EvalMetricsObsGroupCfg()

        self.observations.eval_metrics.leg_reached = ObsTerm(
            func=reached,
            params={
                "source_asset_cfg": SceneEntityCfg("ee_frame"),
                "target_asset_cfg": SceneEntityCfg("square_table_leg_1"),
                "threshold": 0.08,  # Euclidean distance between the end-effector and the target object.
            },
        )
        # TODO: measure per-contact tangential forces and compute the wrenches to decide if a grasp is stable ot not. Currently, the decision is done purely based on the force [N] which is not optimal.
        # the threshold for each grasp type were found by doing the following steps:
        # 1) recording 5-10 min. grasping attemps/grasps.
        # 2) Calculate 1d-histogram and k-means over the data to find meaningful separations.
        # The following grasp classification is purely based on the measured force between both fingers and the target object.
        # TODO: Improve the thresholds by including an estimate about the grasp area.
        self.observations.eval_metrics.grasp_analysis = ObsTerm(
            func=grasp_analysis,
            params={
                "contact_sensors_asset_cfgs": {
                    "contact_forces_left_finger_asset_cfg": SceneEntityCfg(
                        "contact_forces_left_finger"
                    ),
                    "contact_forces_right_finger_asset_cfg": SceneEntityCfg(
                        "contact_forces_right_finger"
                    ),
                },
                "ee_frame_asset_cfg": SceneEntityCfg("ee_frame"),
                "object_asset_cfg": SceneEntityCfg("square_table_leg_1"),
                "reach_threshold": 0.08,
                "very_weak_grasp_threshold": 5.0,  # very weak, might break under any minimal perturbations.
                "weak_grasp_threshold": 15.0,  # it might be strong enough, but the grasp area might differ.
                "strong_grasp_threshold": 26.0,  # includes strong enough grasps, but the grasp area might differ.
            },
        )
        self.observations.eval_metrics.leg_lifted = ObsTerm(
            func=lifted,
            params={
                "contact_sensors_asset_cfgs": {
                    "contact_forces_left_finger_asset_cfg": SceneEntityCfg(
                        "contact_forces_left_finger"
                    ),
                    "contact_forces_right_finger_asset_cfg": SceneEntityCfg(
                        "contact_forces_right_finger"
                    ),
                },
                "ee_frame_asset_cfg": SceneEntityCfg("ee_frame"),
                "object_asset_cfg": SceneEntityCfg("square_table_leg_1"),
                "reach_threshold": 0.08,
                "very_weak_grasp_threshold": 5.0,  # very weak, might break under any minimal perturbations.
                "weak_grasp_threshold": 15.0,  # it might be strong enough, but the grasp area might differ.
                "strong_grasp_threshold": 26.0,  # includes strong enough grasps, but the grasp area might differ.
                "lift_threshold": 0.02,  # this threshold value is only valid if the object is grasped as well.
            },
        )
        self.observations.eval_metrics.leg_inserted = ObsTerm(
            func=inserted,
            params={
                "object_tip_pos_asset_cfg": SceneEntityCfg(
                    "square_table_leg1_target_positions_frame"
                ),
                "target_pos_asset_cfg": SceneEntityCfg(
                    "square_table_top_target_positions_frame"
                ),
                "threshold": 0.0185,  # Euclidean-distance for which the leg is in the assembly slot, but can be skewed (not perfectly vertically aligned).
            },
        )
        self.observations.eval_metrics.leg_aligned = ObsTerm(
            func=z_axis_aligned,
            params={
                "object_tip_pos_asset_cfg": SceneEntityCfg(
                    "square_table_leg1_target_positions_frame"
                ),
                "target_pos_asset_cfg": SceneEntityCfg(
                    "square_table_top_target_positions_frame"
                ),
                "threshold": 0.15,  # rotational difference along z-axis in radians (8.594367 degrees)
            },
        )

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
                usd_path=assets.get_absolute_path("obstacle_front.usd"),
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
                usd_path=assets.get_absolute_path("obstacle_side.usd"),
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
                usd_path=assets.get_absolute_path("obstacle_side.usd"),
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
                usd_path=assets.get_absolute_path("square_table_top.usd"),
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
                usd_path=assets.get_absolute_path("square_table_leg1.usd"),
                rigid_props=rigid_body_properties,
                collision_props=collision_props_table_parts,
                mass_props=leg_mass,
                semantic_tags=[("class", "square_table_leg_1")],
            ),
        )

        # contact sensors for both franka fingers, useful for grasp analysis
        self.scene.contact_forces_right_finger = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
            update_period=0.0,
            history_length=0,
            debug_vis=False,
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/SquareTable_Leg_1/square_table_leg1"
            ],
        )
        self.scene.contact_forces_left_finger = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
            update_period=0.0,
            history_length=0,
            debug_vis=False,
            filter_prim_paths_expr=[
                "{ENV_REGEX_NS}/SquareTable_Leg_1/square_table_leg1"
            ],
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
