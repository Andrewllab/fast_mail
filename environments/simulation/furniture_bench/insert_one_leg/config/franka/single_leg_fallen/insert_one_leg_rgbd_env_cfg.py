# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Literal
import torch

from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

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
from ....config import camera_params
from .....insert_one_leg import assets

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


def get_camera_parameters(
    file_name: str,
    parameter_type: str = Literal["intrinsics", "extrinsics"],
    height=480,
    width=640,
):
    """Read camera parameters from yaml file.
    The intrinsics parameters depends on the resolution.
    Args:
        file_path: the path to the yaml-file of the camera parameters
        parameter_type: the type of the camera parameters. Supported types: extrinsics and intrinsics. The intrinsics parameters depend on the resolution.

    Returns:
        Tensor shape is (9,) for the intrinsics and dict("pos": (3,), "rot_quat": (4,)) for the extrinsics.
    """

    cfg = camera_params.load(file_name)

    match parameter_type:

        case "extrinsics":
            extrinsic_params = cfg["extrinsics"]
            extrinsic_params = torch.tensor(extrinsic_params).reshape(
                4, 4
            )  # create a homogenious matrix from the flattened vector
            # Split the homogenious matrix into position and rotation (as a quaternion) parts.
            # This way, the environment config files remains cleaner.
            camera_parameters = {"pos": None, "rot": None}
            camera_parameters["pos"], camera_parameters["rot"] = math_utils.unmake_pose(
                extrinsic_params
            )
            camera_parameters["rot"] = math_utils.quat_from_matrix(
                camera_parameters["rot"]
            )
            camera_parameters["rot"] = math_utils.quat_unique(
                camera_parameters["rot"]
            )  # orientation representation as a quaternion is not unique (+q, -q)

        case "intrinsics":
            camera_parameters = cfg["intrinsics"]["resolution"][
                f"{height}x{width}"
            ]  # IsaacLab already expects a flattened list

        case unsupported:
            raise ValueError(
                f"Required data type '{unsupported}' is not supported. Supported data types: extrinsics, intrinsics."
            )

    return camera_parameters


def extract_camera_parameters(
    env,
    camera_name: str,
    parameter_type: Literal["intrinsics", "extrinsics"],
    convention: Literal["world", "opengl", "ros"],
):
    """Extract intrinsic/extrinsic camera parameters and preprocess them for recording."""

    match parameter_type:
        case "intrinsics":
            cam = env.scene[camera_name]
            cam_data = cam.data  # triggers update
            camera_parameters = cam_data.intrinsic_matrices.clone()

        case "extrinsics":
            cam = env.scene[camera_name]
            cam_data = cam.data  # triggers update
            pos_world_frame = cam_data.pos_w.clone()
            match convention:
                case "world":
                    rot = cam_data.quat_w_world.clone()
                case "opengl":
                    rot = cam_data.quat_w_opengl.clone()
                case "ros":
                    rot = cam_data.quat_w_ros.clone()
            rot_matrix = math_utils.matrix_from_quat(rot)
            # create a pose as a homogenious matrix
            camera_parameters = math_utils.make_pose(
                pos=pos_world_frame, rot=rot_matrix
            )

        case _:
            raise ValueError(
                f"Invalid parameter_type '{parameter_type}'. Expected 'intrinsics' or 'extrinsics'."
            )

    return camera_parameters


@configclass
class CameraObsGroupCfg(ObsGroup):
    """An observation group acting as a container for all relevant information per camera."""

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


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

        # Set the reward function as a termination term (and not as a reward) as we currently do only imitation learning
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
        self.scene.robot = assets.franka.FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Set cameras

        # Gripper camera
        gripper_cam_intrinsics_matrix = get_camera_parameters(
            file_name="realsense_d405.yaml",
            parameter_type="intrinsics",
            height=480,
            width=640,
        )
        gripper_cam_extrinsics_matrix = get_camera_parameters(
            file_name="realsense_d405.yaml", parameter_type="extrinsics"
        )
        self.scene.gripper_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/gripper_cam",
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=gripper_cam_intrinsics_matrix,
                height=480,
                width=640,
                clipping_range=(0.01, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=gripper_cam_extrinsics_matrix["pos"],
                rot=gripper_cam_extrinsics_matrix["rot"],
                convention="ros",
            ),
        )

        # Static front right camera
        static_front_right_cam_intrinsics_matrix = get_camera_parameters(
            file_name="static_right_realsense_d435.yaml",
            parameter_type="intrinsics",
            height=480,
            width=640,
        )
        static_front_right_cam_extrinsics_matrix = get_camera_parameters(
            file_name="static_right_realsense_d435.yaml", parameter_type="extrinsics"
        )
        self.scene.front_right_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_right_cam",
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=static_front_right_cam_intrinsics_matrix,
                height=480,
                width=640,
                clipping_range=(0.01, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=static_front_right_cam_extrinsics_matrix["pos"],
                rot=static_front_right_cam_extrinsics_matrix["rot"],
                convention="ros",
            ),
        )

        # Static front left camera
        static_front_left_cam_intrinsics_matrix = get_camera_parameters(
            file_name="static_left_realsense_d435.yaml",
            parameter_type="intrinsics",
            height=480,
            width=640,
        )
        rel_static_front_left_cam_extrinsics_matrix = get_camera_parameters(
            file_name="static_left_realsense_d435.yaml", parameter_type="extrinsics"
        )
        # currently, the left cam extrinsics are relative to the right (leader) camera.
        # Thus, compute the world-frame pose of the left cam using the relative pose to the right (leader) camera and the right (leader) camera pose in world frame.
        static_front_left_cam_pos, static_front_left_cam_rot = (
            math_utils.combine_frame_transforms(
                static_front_right_cam_extrinsics_matrix["pos"],
                static_front_right_cam_extrinsics_matrix["rot"],
                rel_static_front_left_cam_extrinsics_matrix["pos"],
                rel_static_front_left_cam_extrinsics_matrix["rot"],
            )
        )
        self.scene.front_left_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_left_cam",
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=static_front_left_cam_intrinsics_matrix,
                height=480,
                width=640,
                clipping_range=(0.01, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=static_front_left_cam_pos,
                rot=static_front_left_cam_rot,
                convention="ros",
            ),
        )

        # self.scene.static_camera_back = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/static_camera_back",
        #     height=480,
        #     width=640,
        #     data_types=["rgb", "distance_to_image_plane"],
        #     spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        #         intrinsic_matrix=static_left_cam_intrinsics_matrix, # just take intrinsics from any camera as this camera is actually not used in the real world scene.
        #         height=480,
        #         width=640,
        #         clipping_range=(0.01, 1.0e5),
        #     ),
        #     offset=CameraCfg.OffsetCfg(
        #         pos=(-0.05, -0.2, 0.45), rot=(0.67719, 0.44454, -0.25974, -0.52568), convention="opengl" # hard-coded pose as this camera is actually not used in the real world.

        #     ),
        # )

        # Add all necessary per-camera observation groups
        # Gripper-cam observations
        self.observations.gripper_cam = CameraObsGroupCfg()
        self.observations.gripper_cam.rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("gripper_cam"),
                "data_type": "rgb",
                "convert_perspective_to_orthogonal": False,
                "normalize": False,
            },
        )
        # add orthogonal ("distance_to_image_plane") depth images to obs
        # https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html#depth-and-distances
        self.observations.gripper_cam.depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("gripper_cam"),
                "data_type": "distance_to_image_plane",
                "convert_perspective_to_orthogonal": True,
                "normalize": False,
            },
        )
        # add extrinsics parameters of the gripper cam as observations since they change dynamically
        self.observations.gripper_cam.extrinsics = ObsTerm(
            func=extract_camera_parameters,
            params={
                "camera_name": "gripper_cam",
                "parameter_type": "extrinsics",
                "convention": "ros",
            },
        )

        # Front right static cam observations
        self.observations.front_right_cam = CameraObsGroupCfg()
        self.observations.front_right_cam.rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("front_right_cam"),
                "data_type": "rgb",
                "convert_perspective_to_orthogonal": False,
                "normalize": False,
            },
        )
        # add orthogonal ("distance_to_image_plane") depth images to obs
        # https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html#depth-and-distances
        self.observations.front_right_cam.depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("front_right_cam"),
                "data_type": "distance_to_image_plane",
                "convert_perspective_to_orthogonal": True,
                "normalize": False,
            },
        )

        # Front left static cam observations
        self.observations.front_left_cam = CameraObsGroupCfg()
        self.observations.front_left_cam.rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("front_left_cam"),
                "data_type": "rgb",
                "convert_perspective_to_orthogonal": False,
                "normalize": False,
            },
        )
        # add orthogonal ("distance_to_image_plane") depth images to obs
        # https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html#depth-and-distances
        self.observations.front_left_cam.depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("front_left_cam"),
                "data_type": "distance_to_image_plane",
                "convert_perspective_to_orthogonal": True,
                "normalize": False,
            },
        )

        # Back static cam observations
        # self.observations.back_cam = CameraObsGroupCfg()
        # self.observations.back_cam.rgb = ObsTerm(
        #     func=mdp.image,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("static_camera_back"),
        #         "data_type": "rgb",
        #         "convert_perspective_to_orthogonal": False,
        #         "normalize": False,
        #     },
        # )
        # self.observations.back_cam.depth = ObsTerm(
        #     func=mdp.image,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("static_camera_back"),
        #         "data_type": "distance_to_image_plane",
        #         "convert_perspective_to_orthogonal": True,
        #         "normalize": False,
        #     },
        # )

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

        self.rerender_on_reset = True

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]
        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for joint control for the specific robot type (franka)
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
        obstacle_mass = sim_utils.MassPropertiesCfg(mass=0.065 * 10)
        table_top_mass = sim_utils.MassPropertiesCfg(mass=0.167 * 10)
        leg_mass = sim_utils.MassPropertiesCfg(mass=0.036 * 10)

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
                        # pos=[0.0, 0.0, 0.1034], # corresponds to the middle point of the gripper knobs
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
