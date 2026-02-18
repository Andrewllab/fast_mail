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


def pointcloud_sort(points, other_tensors=[]):
    # sort by z (lowest priority)
    sort_keys = points[:, 2].argsort(stable=True)
    points = points[sort_keys]
    for i in range(len(other_tensors)):
        other_tensors[i] = other_tensors[i][sort_keys]

    # then by y
    sort_keys = points[:, 1].argsort(stable=True)
    points = points[sort_keys]
    for i in range(len(other_tensors)):
        other_tensors[i] = other_tensors[i][sort_keys]

    # finally by x (highest priority)
    sort_keys = points[:, 0].argsort(stable=True)
    points = points[sort_keys]
    for i in range(len(other_tensors)):
        other_tensors[i] = other_tensors[i][sort_keys]

    return points, other_tensors


def main():
    dataset_path = "/mnt/SmallSandwich/3d-sim2real_prepreprocessed_/robocasa/kitchen_pnp_seg_pointclouds"
    dataset_files = glob.glob(f"{dataset_path}/*.hdf5")
    dataset_files.sort()

    sigmas = []

    vis_options = dict(point_size=5, line_width=2, show_ui=True)

    for dataset_path in dataset_files:
        with h5py.File(dataset_path, "r") as f:
            data = recursive_hdf5_to_dict(f)
        data = unpack_nested_from_storage(data)

        track_tool_pcd = data["obs"]["tool_points"]
        mask_tool_pcd = data["obs"]["mask_tool_points"]

        for step in range(track_tool_pcd["points"].shape[0]):
            track_tool_points = track_tool_pcd["points"][step][:, 0]
            mask_tool_points = mask_tool_pcd["points"][step]

            track_tool_points, [track_features, track_colors] = pointcloud_sort(
                track_tool_points,
                [
                    track_tool_pcd["features"][step],
                    track_tool_pcd["colors"][step],
                ],
            )
            mask_tool_points, [mask_features, mask_colors] = pointcloud_sort(
                mask_tool_points,
                [
                    mask_tool_pcd["features"][step],
                    mask_tool_pcd["colors"][step],
                ],
            )
            pass


if __name__ == "__main__":
    main()
