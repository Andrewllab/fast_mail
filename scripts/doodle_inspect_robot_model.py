from pathlib import Path

import rootutils
import torch
import torchcontrol as toco
from tensordict import TensorDict

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from utils.math import (
    convert_quat,
    subtract_frame_transforms,
    subtract_frame_transforms2,
)

if __name__ == "__main__":

    torch.set_printoptions(linewidth=120)

    demos_root_path = Path.home() / "datasets/3d-sim2real/insert_one_leg/isaac_sim/"

    demo_filepath = demos_root_path / "metaquest3_2025-07-24/insert_one_leg_9.hdf5"
    # demo_filepath = (
    #     demos_root_path
    #     / "rotational_movements_along_one_axis_keyboard_ik_abs/rotational_movements_along_one_axis_keyboard_ik_abs.hdf5"
    # )

    # demo_filepath = (
    #     Path.home()
    #     / "lsdf/projects/3d-sim2real/demonstrations/insert_one_leg/isaac_sim/debugging/open_loop_replay_pre_recorded_sim_demos/rotational_movements_along_one_axis_keyboard_ik_abs.hdf5"
    # )

    urdf_filepath = Path.home() / (
        "repos/novometis/polymetis/data/franka_panda/panda_arm.urdf"
    )
    ee_link_name = "panda_link8"

    traj = TensorDict.from_h5(str(demo_filepath))
    traj = traj["data", "demo_0"]
    joint_pos = traj["obs", "proprioception", "joint_pos"]
    ee_pos_world = traj["obs", "proprioception", "eef_pos_w"]
    ee_quat_world = traj["obs", "proprioception", "eef_quat_w"]

    robot_model = toco.models.RobotModelPinocchio(str(urdf_filepath), ee_link_name)

    first_ee_pos_base, first_ee_quat_base = robot_model.forward_kinematics(joint_pos[0])
    first_ee_quat_base = convert_quat(first_ee_quat_base, to="wxyz")

    base_pos_world, base_quat_world = subtract_frame_transforms2(
        t12=first_ee_pos_base.unsqueeze(0),
        q12=first_ee_quat_base.unsqueeze(0),
        t02=ee_pos_world[:1],
        q02=ee_quat_world[:1],
    )

    base_pos_world_repeat = base_pos_world.repeat_interleave(len(ee_pos_world), dim=0)
    base_quat_world_repeat = base_quat_world.repeat_interleave(len(joint_pos), dim=0)

    demo_ee_pos_base, demo_ee_quat_base = subtract_frame_transforms(
        t01=base_pos_world_repeat,
        q01=base_quat_world_repeat,
        t02=ee_pos_world,
        q02=ee_quat_world,
    )
    demo_ee_pose_base = torch.cat((demo_ee_pos_base, demo_ee_quat_base), dim=-1)

    model_ee_pose_base = []
    for joint_pos_i in joint_pos:
        model_ee_pos_base, model_ee_quat_base = robot_model.forward_kinematics(
            joint_pos_i
        )
        model_ee_quat_base = convert_quat(model_ee_quat_base, to="wxyz")
        model_ee_pose_base.append(
            torch.cat((model_ee_pos_base, model_ee_quat_base), dim=-1)
        )

    model_ee_pose_base = torch.stack(model_ee_pose_base)

    success = torch.allclose(model_ee_pose_base, demo_ee_pose_base)

    print(f"{joint_pos[0]=}")
    print(f"demo_ee_pose[0]={torch.cat((ee_pos_world[0], ee_quat_world[0]))}")

    for i in range(9):
        link = f"panda_link{i}"
        link_pos, link_quat = robot_model.forward_kinematics(joint_pos[0], link)
        link_quat = convert_quat(link_quat, to="wxyz")
        link_pose = torch.cat((link_pos, link_quat))
        print(f"{link}[0]={link_pose}")
        print(f"{link}[0]={link_pose}")
