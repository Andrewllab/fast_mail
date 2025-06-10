# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from pathlib import Path
from collections import defaultdict
from typing import Literal
import torch 
import yaml

from isaaclab.assets import RigidObjectCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab_tasks.manager_based.manipulation.stack import mdp
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
import isaaclab.utils.math as math_utils

from ...furniture_bench_table_env_cfg import InsertOneLegEnvCfg
from ...mdp.events import reset_table_parts_poses, randomize_object_position, randomize_light_intensity
from ...mdp.terminations import success

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from ...assets.franka import FRANKA_PANDA_CFG

BASE_PATH = Path(__file__).parent.parent.parent


def get_camera_parameters(file_path: str, parameter_type: str = Literal["intrinsics", "extrinsics"], height=480, width=640):
    """Read camera parameters from yaml file. 
    The intrinsics parameters depends on the resolution.
    Args:
        file_path: the path to the yaml-file of the camera parameters
        parameter_type: the type of the camera parameters. Supported types: extrinsics and intrinsics. The intrinsics parameters depend on the resolution.

    Returns: 
        Tensor shape is (9,) for the intrinsics and dict("pos": (3,), "rot_quat": (4,)) for the extrinsics.
    """

    with open(file_path, "r") as f:
        cfg = yaml.safe_load(f)

    match parameter_type:

        case "extrinsics":
            extrinsic_params = cfg["extrinsics"]
            extrinsic_params = torch.tensor(extrinsic_params).reshape(4, 4) # create a homogenious matrix from the flattened vector
            # Split the homogenious matrix into position and rotation (as a quaternion) parts.
            # This way, the environment config files remains cleaner.
            camera_params = {"pos": None,
                             "rot": None}
            camera_params["pos"], camera_params["rot"] = math_utils.unmake_pose(extrinsic_params)
            camera_params["rot"] = math_utils.quat_from_matrix(camera_params["rot"])
            camera_params["rot"] = math_utils.quat_unique(camera_params["rot"]) # orientation representation as a quaternion is not unique (+q, -q)

        case "intrinsics":
            camera_params = cfg["intrinsics"]["resolution"][f"{height}x{width}"] # IsaacLab already expects a flattened list

        case unsupported:
            raise ValueError(f"Required data type '{unsupported}' is not supported. Supported data types: extrinsics, intrinsics.")

    return camera_params


def extract_camera_parameters(env, camera_name: str, parameter_type: Literal["intrinsics", "extrinsics"]):
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
            # orientation_world_frame = cam_data.quat_w_world.clone()
            rot_world_frame = cam_data.quat_w_world.clone()
            rot_world_frame_matrix = math_utils.matrix_from_quat(rot_world_frame)
            # create a pose as a homogenious matrix
            camera_parameters = math_utils.make_pose(pos=pos_world_frame, rot=rot_world_frame_matrix)

        case _:
            raise ValueError(f"Invalid parameter_type '{parameter_type}'. Expected 'intrinsics' or 'extrinsics'.")

    return camera_parameters


@configclass
class CameraObsGroupCfg(ObsGroup):
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
            "default_pose": [0.1995, -0.2052, -0.2379, -2.5128, 0.0027, 2.3185, -0.6007, 0.0400, 0.0400], # an approximation of the real-world initial franka-pose
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
                "square_table_top": {"position": [0.515, 0.092, 0.01], "orientation": [0, 0, 0.7071068, 0.7071068]},
            }
        },
    )

    randomize_light_intensity = EventTerm(
        func=randomize_light_intensity,
        mode="reset",
        params={
            "intensity_range": (20000, 50000),
            "asset_cfgs": [SceneEntityCfg("background_light"), SceneEntityCfg("front_light")],
        },
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
        self.terminations.success = DoneTerm(func=success, 
                                params={"xy_threshold": 0.0004, "height_threshold": 0.00025}, # figured out through a lot of experimentation 
                                time_out=True) 

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]

        # Set cameras
        # TODO: Find a way to wrap all cameras into one data structure. 
        # When storing them into a dict (e.g. self.scene.cameras["gripper_camera"]), IsaacLab throws an error: "ValueError: Unknown asset config type for cameras"

        # Gripper camera
        wrist_cam_intrinsics_matrix = get_camera_parameters(file_path=os.path.join(BASE_PATH, "config/camera_params/realsense_d405.yaml"), parameter_type="intrinsics", height=480, width=640)
        wrist_cam_extrinsics_matrix = get_camera_parameters(file_path=os.path.join(BASE_PATH, "config/camera_params/realsense_d405.yaml"), parameter_type="extrinsics")
        self.scene.gripper_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/gripper_cam",
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=wrist_cam_intrinsics_matrix,
                width=640,
                height=480,
                clipping_range=(0.01, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=wrist_cam_extrinsics_matrix["pos"], 
                rot=wrist_cam_extrinsics_matrix['rot'], 
                convention="opengl" # manually adapted since eef of the real robot not known
            ),
        )

        # Static right camera
        static_right_cam_intrinsics_matrix = get_camera_parameters(file_path=os.path.join(BASE_PATH, "config/camera_params/static_right_realsense_d435.yaml"), parameter_type="intrinsics", height=480, width=640)
        static_right_cam_extrinsics_matrix = get_camera_parameters(file_path=os.path.join(BASE_PATH, "config/camera_params/static_right_realsense_d435.yaml"), parameter_type="extrinsics")
        self.scene.right_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/right_cam",
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=static_right_cam_intrinsics_matrix,
                height=480,
                width=640,
                clipping_range=(0.01, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos = static_right_cam_extrinsics_matrix["pos"],
                rot = static_right_cam_extrinsics_matrix["rot"],
                convention="ros",
            )
        )

        # Static left camera
        static_left_cam_intrinsics_matrix = get_camera_parameters(file_path=os.path.join(BASE_PATH, "config/camera_params/static_left_realsense_d435.yaml"), parameter_type="intrinsics", height=480, width=640)
        rel_static_left_cam_extrinsics_matrix = get_camera_parameters(file_path=os.path.join(BASE_PATH, "config/camera_params/static_left_realsense_d435.yaml"), parameter_type="extrinsics")
        # currently, the left cam extrinsics are relative to the right (leader) camera.
        # Thus, compute the world-frame pose of the left cam using the relative pose to the right (leader) camera and the right (leader) camera pose in world frame.
        static_left_cam_pos, stat_left_cam_rot = math_utils.combine_frame_transforms(
            static_right_cam_extrinsics_matrix["pos"],
            static_right_cam_extrinsics_matrix["rot"],
            rel_static_left_cam_extrinsics_matrix["pos"],
            rel_static_left_cam_extrinsics_matrix["rot"],
        )
        self.scene.left_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/left_cam",
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=static_left_cam_intrinsics_matrix,
                height=480,
                width=640,
                clipping_range=(0.01, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos = static_left_cam_pos,
                rot = stat_left_cam_rot,
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
        self.observations.gripper_cam.extrinsics = ObsTerm(func=extract_camera_parameters, 
                                                                 params={
                                                                     "camera_name": "gripper_cam",
                                                                     "parameter_type": "extrinsics"
                                                                    }
                                                                )

        # Left static cam observations
        self.observations.left_cam = CameraObsGroupCfg()
        self.observations.left_cam.rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_cam"),
                "data_type": "rgb",
                "convert_perspective_to_orthogonal": False,
                "normalize": False,
            },
        )
        # add orthogonal ("distance_to_image_plane") depth images to obs
        # https://isaac-sim.github.io/IsaacLab/main/source/overview/core-concepts/sensors/camera.html#depth-and-distances
        self.observations.left_cam.depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_cam"),
                "data_type": "distance_to_image_plane",
                "convert_perspective_to_orthogonal": True,
                "normalize": False,
            },
        )

        # Right static cam observations
        self.observations.right_cam = CameraObsGroupCfg()
        self.observations.right_cam.rgb = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("right_cam"),
                "data_type": "rgb",
                "convert_perspective_to_orthogonal": False,
                "normalize": False,
            },
        )
        self.observations.right_cam.depth = ObsTerm(
            func=mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("right_cam"),
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
        self.rerender_on_reset = True

        # Add semantics to table
        self.scene.table.spawn.semantic_tags = [("class", "table")]
        # Add semantics to ground
        self.scene.plane.semantic_tags = [("class", "ground")]

        # Set actions for joint control for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )


        # Setup all static and dynamic objects, including their properties
        rigid_body_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=64,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=3666.0,
            max_depenetration_velocity=5000.0,
            disable_gravity=False,
            max_contact_impulse=1000.0,
            linear_damping=0.0,
            angular_damping=0.0,
        )

        collision_props_table_parts=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.005,
            rest_offset=0.0,
        )

        # increasing the actual object's masses seems to imporve simulation stability.
        # Too high difference between the franka's arm mass and a manipulated object's mass is unstable.
        # E.g., between the object and the franka arm there are interpenetrations, also the object jitters during manipulation.
        obstacle_mass = sim_utils.MassPropertiesCfg(
            mass=0.065 * 10
        )
        table_top_mass = sim_utils.MassPropertiesCfg(
            mass=0.167 * 10
        )
        leg_mass = sim_utils.MassPropertiesCfg(
            mass=0.036 * 10
        )

        # Obstacles 
        static_body_properties = RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
        )
        #  front 
        self.scene.obstacle_front = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObstacleFront",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.61, 0, 0.01], rot=[0.707, 0, 0, 0.707]),
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.535, 0.185, 0.01], rot=[0.707, 0, 0, 0.707]),
            spawn=UsdFileCfg(
                usd_path=os.path.join(BASE_PATH, "assets/obstacle_side.usd"),
                rigid_props=static_body_properties,
                mass_props=obstacle_mass,
            ),
        )
        # right side
        self.scene.obstacle_right_side = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/ObstacleRightSide",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.535, -0.185, 0.01], rot=[0.707, 0, 0, 0.707]),
            spawn=UsdFileCfg(
                usd_path=os.path.join(BASE_PATH, "assets/obstacle_side.usd"),
                rigid_props=static_body_properties,
                mass_props=obstacle_mass,
            ),
        )

        # square table parts 
        self.scene.square_table_top = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SquareTable_Top",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.515, 0.092, 0.05], rot=[0, 0, 0.7071068, 0.7071068]),
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
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3139, -0.0475, 0.0600], rot=[0, 0, 0.7071068, 0.7071068]),
            spawn=UsdFileCfg(
                usd_path=os.path.join(BASE_PATH, "assets/square_table_leg1.usd"),
                rigid_props=rigid_body_properties,
                collision_props=collision_props_table_parts,
                mass_props=leg_mass,
                semantic_tags=[("class", "square_table_leg_1")],
            ),
        )

        self.scene.square_table_leg_2 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SquareTable_Leg_2",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2775, 0.0723, 0.0600], rot=[0, 0, 0.7071068, 0.7071068]),
            spawn=UsdFileCfg(
                usd_path=os.path.join(BASE_PATH, "assets/square_table_leg2.usd"),
                rigid_props=rigid_body_properties,
                collision_props=collision_props_table_parts,
                mass_props=leg_mass,
                semantic_tags=[("class", "square_table_leg_2")],
            ),
        )

        self.scene.square_table_leg_3 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SquareTable_Leg_3",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2922, -0.1470, 0.0600], rot=[0, 0, 0.7071068, 0.7071068]), 
            spawn=UsdFileCfg(
                usd_path=os.path.join(BASE_PATH, "assets/square_table_leg3.usd"),
                rigid_props=rigid_body_properties,
                collision_props=collision_props_table_parts,
                mass_props=leg_mass,
                semantic_tags=[("class", "square_table_leg_3")],
            ),
        )

        self.scene.square_table_leg_4 = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/SquareTable_Leg_4",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3309, -0.2494,  0.0600], rot=[0, 0, 0.7071068, 0.7071068]),
            spawn=UsdFileCfg(
                usd_path=os.path.join(BASE_PATH, "assets/square_table_leg4.usd"),
                rigid_props=rigid_body_properties,
                collision_props=collision_props_table_parts,
                mass_props=leg_mass,
                semantic_tags=[("class", "square_table_leg_4")],
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
                        pos=[0.0, 0.0, 0.1034], # corresponds to the middle point of the gripper knobs
                    ),
                ),
            ],
        )

        # reward relevant markers 
        targets_marker_cfg = FRAME_MARKER_CFG.copy()
        targets_marker_cfg.markers["frame"].scale = (0.001, 0.001, 0.001)
        targets_marker_cfg.prim_path = "/Visuals/SquareTableTopTargetFrameTransformer"
        self.scene.square_table_top_target_positions_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/SquareTable_Top/square_table_top", # make sure that the path is showing to the rigid-body prim which is not necessarily the root prim
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
            prim_path="{ENV_REGEX_NS}/SquareTable_Leg_1/square_table_leg1", # make sure that the path is showing to the rigid-body prim which is not necessarily the root prim
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
        self.scene.square_table_leg2_target_positions_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/SquareTable_Leg_2/square_table_leg2", # make sure that the path is showing to the rigid-body prim which is not necessarily the root prim
            debug_vis=True,
            visualizer_cfg=targets_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/SquareTable_Leg_2/square_table_leg2",
                    name="square_table_leg2_tip",
                    offset=OffsetCfg(
                        pos=(0.0, -0.0568, 0.0), # it is not a bug, this leg is a bit longer
                    ),
                ),
            ],
        )
        self.scene.square_table_leg3_target_positions_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/SquareTable_Leg_3/square_table_leg3", # make sure that the path is showing to the rigid-body prim which is not necessarily the root prim
            debug_vis=True,
            visualizer_cfg=targets_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/SquareTable_Leg_3/square_table_leg3",
                    name="square_table_leg3_tip",
                    offset=OffsetCfg(
                        pos=(0.0, -0.0565, 0.0),
                    ),
                ),
            ],
        )
        self.scene.square_table_leg4_target_positions_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/SquareTable_Leg_4/square_table_leg4", # make sure that the path is showing to the rigid-body prim which is not necessarily the root prim
            debug_vis=True,
            visualizer_cfg=targets_marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/SquareTable_Leg_4/square_table_leg4",
                    name="square_table_leg4_tip",
                    offset=OffsetCfg(
                        pos=(0.0, -0.0565, 0.0),
                    ),
                ),
            ],
        )
