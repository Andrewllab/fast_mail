import glob
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import open3d as o3d
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.hdf5_utils import recursive_hdf5_to_dict
from utils.nested import unpack_nested_from_storage


def main():
    dataset_path = "/mnt/SmallSandwich/3d-sim2real_mujoco_preprocessed/robocasa/kitchen_pnp_seg_pointclouds"
    dataset_files = glob.glob(f"{dataset_path}/*.hdf5")
    dataset_files.sort()

    num_target_points_list = []
    num_tool_points_list = []

    for dataset_path in dataset_files:
        with h5py.File(dataset_path, "r") as f:
            data = recursive_hdf5_to_dict(f)
        data = unpack_nested_from_storage(data)

        tool_points = data["obs"]["tool_points"]["points"]
        target_points = data["obs"]["target_points"]["points"]

        num_target_points_list.extend(
            torch.diff(target_points.offsets()).cpu().tolist()
        )
        num_tool_points_list.extend(torch.diff(tool_points.offsets()).cpu().tolist())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].hist(num_target_points_list, bins=50, color="blue", alpha=0.7)
    axes[0].set_title("Number of Target Points per Step")
    axes[0].set_xlabel("Number of Target Points")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(num_tool_points_list, bins=40, color="orange", alpha=0.7)
    axes[1].set_title("Number of Tool Points per Step")
    axes[1].set_xlabel("Number of Tool Points")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
