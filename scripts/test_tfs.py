# fmt: off

import glob
import os
import sys
from itertools import permutations
from pathlib import Path

import h5py
import open3d as o3d
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.hdf5_utils import recursive_hdf5_to_dict
from utils.math import (
    combine_frame_transforms,
    convert_quat,
    quat_box_minus,
    quat_from_axis_angle,
    subtract_frame_transforms,
    subtract_frame_transforms2,
)
from utils.nested import unpack_nested_from_storage


def main():
    dataset_path = "/mnt/SmallSandwich/3d-sim2real/demonstrations/robocasa/single_stage_1440x1080_seg/kitchen_pnp/PnPCounterToStove/pnp_counter_to_stove_preprocessed_1440x1080.hdf5"

    with h5py.File(dataset_path, "r") as f:
        demo_path = "data/demo_1"
        data = f[demo_path]

        # How do we get ee_pos from this without inverting it
        # BASE TO EEF
        base_to_eef_pos = torch.tensor(
            data["obs"]["robot0_base_to_eef_pos"][:], dtype=torch.float32
        )
        base_to_eef_quat = torch.tensor(
            data["obs"]["robot0_base_to_eef_quat"][:], dtype=torch.float32
        )
        base_to_eef_quat_site = torch.tensor(
            data["obs"]["robot0_base_to_eef_quat_site"][:], dtype=torch.float32
        )
        
        # convert
        base_to_eef_quat = convert_quat(base_to_eef_quat, "wxyz")
        base_to_eef_quat_site = convert_quat(base_to_eef_quat_site, "wxyz")

        # Inverse BASE TO EEF
        inv_base_to_eef_pos, inv_base_to_eef_quat = subtract_frame_transforms(
            base_to_eef_pos, base_to_eef_quat
        )
        inv_base_to_eef_pos, inv_base_to_eef_quat_site = subtract_frame_transforms(
            base_to_eef_pos, base_to_eef_quat_site,
        )

        # EEF
        ee_pos = torch.tensor(data["obs"]["robot0_eef_pos"][:], dtype=torch.float32)
        ee_quat = torch.tensor(data["obs"]["robot0_eef_quat"][:], dtype=torch.float32)
        ee_quat_site = torch.tensor(data["obs"]["robot0_eef_quat_site"][:], dtype=torch.float32)

        # convert
        ee_quat = convert_quat(ee_quat, "wxyz")
        ee_quat_site = convert_quat(ee_quat_site, "wxyz")

        # BASE
        base_pos = torch.tensor(data["obs"]["robot0_base_pos"][:], dtype=torch.float32)
        base_quat = torch.tensor(data["obs"]["robot0_base_quat"][:], dtype=torch.float32)
        
        # convert
        base_quat = convert_quat(base_quat, "wxyz")

        # INVERSE BASE
        inv_base_pos, inv_base_quat = subtract_frame_transforms(
            base_pos,
            base_quat,
        )

        # ACTION
        action_dict = data["action_dict"]
        action_pos = torch.tensor(action_dict["abs_pos"][:], dtype=torch.float32)
        action_rot = torch.tensor(action_dict["abs_rot_axis_angle"][:], dtype=torch.float32)
        action_quat = quat_from_axis_angle(action_rot)

        options = [
            # {"pos": action_pos, "quat": action_quat, "type": "action"},
            # {"pos": ee_pos, "quat": ee_quat, "type": "ee"},
            # {"pos": ee_pos, "quat": ee_quat_site, "type": "ee_with_site"},
            {"pos": base_pos, "quat": base_quat, "type": "base"},
            # {"pos": inv_base_pos, "quat": inv_base_quat, "type": "inv_base"},
            {"pos": base_to_eef_pos, "quat": base_to_eef_quat, "type": "base_to_eef"},
            {
                "pos": base_to_eef_pos,
                "quat": base_to_eef_quat_site,
                "type": "base_to_eef_site",
            },
            {
                "pos": inv_base_to_eef_pos,
                "quat": inv_base_to_eef_quat,
                "type": "inv_base_to_eef",
            },
            {
                "pos": inv_base_to_eef_pos,
                "quat": inv_base_to_eef_quat_site,
                "type": "inv_base_to_eef_site",
            },
        ]

        for perm in permutations(options, 2):
            first, second = perm
            print(f"Combining {first['type']} and {second['type']}")
            first_pos = first["pos"]
            first_quat = first["quat"]

            second_pos = second["pos"]
            second_quat = second["quat"]

            v1_pos, v1_quat = combine_frame_transforms(
                first_pos, first_quat, second_pos, second_quat
            )

            v2_pos, v2_quat = combine_frame_transforms(
                second_pos, second_quat, first_pos, first_quat
            )

            v3_pos, v3_quat = subtract_frame_transforms(
                first_pos, first_quat, second_pos, second_quat
            )

            v4_pos, v4_quat = subtract_frame_transforms(
                second_pos, second_quat, first_pos, first_quat
            )

            v5_pos, v5_quat = subtract_frame_transforms2(
                first_pos, first_quat, second_pos, second_quat
            )

            v6_pos, v6_quat = subtract_frame_transforms2(
                first_pos, first_quat, second_pos, second_quat
            )

            print((v1_pos - ee_pos)[0], quat_box_minus(v1_quat, ee_quat_site)[0])
            print((v2_pos - ee_pos)[0], quat_box_minus(v2_quat, ee_quat_site)[0])
            print((v3_pos - ee_pos)[0], quat_box_minus(v3_quat, ee_quat_site)[0])
            print((v4_pos - ee_pos)[0], quat_box_minus(v4_quat, ee_quat_site)[0])
            print((v5_pos - ee_pos)[0], quat_box_minus(v5_quat, ee_quat_site)[0])
            print((v6_pos - ee_pos)[0], quat_box_minus(v6_quat, ee_quat_site)[0])
        pass


if __name__ == "__main__":
    main()
