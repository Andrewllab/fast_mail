import glob
import os
import sys
from pathlib import Path

import h5py
import open3d as o3d

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.hdf5_utils import recursive_hdf5_to_dict
from utils.nested import unpack_nested_from_storage


def main():
    dataset_path = "/mnt/SmallSandwich/3d-sim2real_preprocessed/robocasa/kitchen_pnp_seg_pointclouds"
    dataset_files = glob.glob(f"{dataset_path}/*.hdf5")

    sigmas = []

    vis_options = dict(point_size=5, line_width=2, show_ui=True)

    for dataset_path in dataset_files:
        with h5py.File(dataset_path, "r") as f:
            data = recursive_hdf5_to_dict(f)
        data = unpack_nested_from_storage(data)

        pc_keys = [
            "obs/gripper_points/points",
            "obs/tool_points/points",
            "obs/target_points/points",
            # "obs/base_pos_tf_action",
            # "obs/inv_base_pos_tf_action",
            # "obs/base_to_ee_pos_tf_action",
            # "obs/inv_base_to_ee_pos_tf_action",
        ]
        colors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
            [0.5, 0.5, 0.5],
        ]
        tool_points = data["obs"]["tool_points"]["points"]
        # target_points = data["obs"]["target_points"]["points"]
        # if "gripper_points" in data["obs"]:
        #     gripper_points = data["obs"]["gripper_points"]["points"]
        # gripper_points = data["obs"]["gripper_points"]["points"]

        for step in range(0, tool_points.shape[0], 20):
            print(f"Visualizing step {step}/{tool_points.shape[0]}")
            o3d_pcds = []
            for i, pc_key in enumerate(pc_keys):
                sub_keys = pc_key.split("/")
                value = data
                for sub_key in sub_keys:
                    value = value[sub_key]

                o3d_pcd = o3d.geometry.PointCloud()
                if value.ndim == 3:
                    o3d_pcd.points = o3d.utility.Vector3dVector(value[step])
                elif value.ndim == 4:
                    o3d_pcd.points = o3d.utility.Vector3dVector(value[step][:, 0])

                o3d_pcd.paint_uniform_color(colors[i])
                o3d_pcds.append({"name": pc_key, "geometry": o3d_pcd})

            # o3d.visualization.draw([tool_o3d, target_o3d, gripper_o3d], **vis_options)
            o3d.visualization.draw(o3d_pcds, **vis_options)

        pass


if __name__ == "__main__":
    main()
