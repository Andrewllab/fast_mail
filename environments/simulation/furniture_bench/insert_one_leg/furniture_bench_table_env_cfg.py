# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import os

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from . import mdp

BASE_PATH = os.path.dirname(__file__)

##
# Scene definition
##
@configclass
class FurnitureBenchTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # # Table
    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0.3, 0, -1.01]),
    #     spawn=UsdFileCfg(usd_path=os.path.join(BASE_PATH, "assets/real_world_table_approximation.usd")),
    # )
    # PROBLEM: real_world_table_approximation is better approximation, but MetaQuest3 throws errors.
        # Current workaround: Wrap the original franka table with an appropriate material
        # TODO: Find a way to load this with MetaQuest3.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=os.path.join(BASE_PATH, "assets/real_world_table_approximation_2.usd")),
    )

    # Background
    background = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Background",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-0.5, 0, 0.0]),
        spawn=UsdFileCfg(usd_path=os.path.join(BASE_PATH, "assets/black_background_flattened.usd")),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    # should approximate the light behind Frankas P3 and P4
    background_light = AssetBaseCfg(
        prim_path="/World/background_light",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.7, 2.5]),
        spawn=sim_utils.SphereLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=50000.0,
        ),
    )
    # should approximate the light in front of Frankas P3 and P4
    front_light = AssetBaseCfg(
        prim_path="/World/front_light",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[3.16, 0.2, 2.5]),
        spawn=sim_utils.SphereLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=25000.0,
        ),
    )

##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        joint_pos = ObsTerm(func=mdp.joint_pos) # all joint positions except the grippers
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel) # all realtive joint positions except the grippers
        joint_vel = ObsTerm(func=mdp.joint_vel) # all joint velocities except the grippers
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel) # all relative joint velocities except the grippers
        eef_pos = ObsTerm(func=mdp.ee_frame_pos) # ee_frame position w.r.t world frame (which is identical with the robot base frame) 
        eef_quat = ObsTerm(func=mdp.ee_frame_quat) # ee_frame orientation w.r.t. world frame.
        gripper_pos = ObsTerm(func=mdp.gripper_pos) # the degree to which the grippers are closed.

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    proprioception: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class InsertOneLegEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the stacking environment."""

    # Scene settings
    scene: FurnitureBenchTableSceneCfg = FurnitureBenchTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()

    # Unused managers
    commands = None
    rewards = None
    events = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        # correspond to the number of simulation sub-steps between envionment steps
        self.decimation = 3
        self.seed = 42 # set seed here for deterministic environments
        # self.sim.physx.use_gpu = False
        # Make it high to prevent environment reset when collection demonstrations
        # For training, set a lower value
        self.episode_length_s = 20.0
        # simulation settings
        # WARNING: sim.dt != env.step_size/rendering step_size
        # environment step_size = sim.dt / decimation !!!
        self.sim.dt = 0.01  # 100Hz 
        self.sim.render_interval = self.decimation
        # GUI viewer perspective location
        # self.viewer.eye = (0.82762, 0.24943, 0.28195) # suitable for analyzing the area near the table assembly slots
        self.viewer.eye = (1.67569, 0.51437, 0.68258) # suitable when recording videos for validation
        # GUI viewer focus point
        # self.viewer.lookat = (0.0, 0.0, -0.2) # suitable for analyzing the area near the table assembly slots
        self.viewer.lookat = (0.0, 0.0, 0.15) # suitable when recording videos for validation

        # inspired from https://github.com/isaac-sim/IsaacLab/blob/3c3103f637f8193a363c4774781089baa19d0465/source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py#L101
        # General solver recommendations: https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/simulation-control/physics-settings.html#physics-solver
        self.sim.physx.solver_type = 1  # TGS solver (seems more stable and faster converging than PGS)
        self.sim.physx.bounce_threshold_velocity = 0.2 # m/s
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.max_position_iteration_count = 192 # Important to avoid interpenetration.
        self.sim.max_velocity_iteration_count = 1
        self.sim.gpu_max_rigid_contact_count = 2**30
        self.sim.gpu_max_rigid_patch_count = 2**30
        self.sim.gpu_max_num_partitions = 1 # Important for stable simulation.

        # The following fricition values seems to be the best trade-off between reaslism and simulation stability gains.
        # Increasing both friction types leads to more stable simulation (in some edge-cases, e.g. when twisting the leg in the assembly slot further than necessary), but the objects starts become too sticky, which increases the sim-2-real gap! 
        # The following MaterialCfg configures the defaults physic's material that PhyX is going to use per-default for all objects that don't have pre-defined physic's material
        # In this case, this affects all environment assets. 
        # If different physic's properties are desired, consider using the pre-defined event function "bind_physics_materials" as an event with mode "pre-startup".
        self.sim.physics_material = RigidBodyMaterialCfg(
            static_friction=0.5, 
            dynamic_friction=0.5,
            compliant_contact_damping=0.5,
            compliant_contact_stiffness=0.5,
        )
