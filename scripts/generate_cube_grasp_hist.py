import argparse
import glob
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)
from utils.math import convert_quat, quaternion_to_euler_xyz

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

parser = argparse.ArgumentParser(
    description="Inject a pre-processed goal embedding into an HDF5 file."
)
parser.add_argument(
    "--folderpath",
    type=str,
    required=True,
    help="Path to the folder containing all .h5 files",
)
parser.add_argument(
    "--out-prefix",
    type=str,
    required=True,
    help="Prefix for the name of the generated heatmaps",
)
args = parser.parse_args()

in_files = glob.glob(os.path.join(args.folderpath, "**", "*.h5"), recursive=True)

border_value = 0.04

gripper_closing_eef_pos_list = []
gripper_closing_eef_quat_list = []

max_workspace = np.array([-np.inf, -np.inf, -np.inf])
min_workspace = np.array([np.inf, np.inf, np.inf])

for in_file in in_files:

    with h5py.File(in_file, "r+") as hf:

        gripper_pos = np.asarray(hf["action"]["gripper_pos"])
        eef_pos = np.asarray(hf["obs"]["proprioception"]["eef_pos"])
        eef_quat = np.asarray(hf["obs"]["proprioception"]["eef_quat"])

        gripper_closing = np.all(
            [gripper_pos[:-1] > border_value, gripper_pos[1:] <= border_value], axis=0
        )

        gripper_closing_indices = np.where(gripper_closing)[0] + 1

        if len(gripper_closing_indices) < 2:
            continue

        gripper_closing_eef_pos = eef_pos[gripper_closing_indices][1]
        gripper_closing_eef_pos_list.append(gripper_closing_eef_pos)

        gripper_closing_eef_quat = eef_quat[gripper_closing_indices][1]
        gripper_closing_eef_quat_list.append(gripper_closing_eef_quat)


gripper_closing_eef_quat_list = np.stack(gripper_closing_eef_quat_list, axis=0)
gripper_closing_eef_euler = torch.stack(
    quaternion_to_euler_xyz(
        convert_quat(torch.tensor(gripper_closing_eef_quat_list), to="wxyz")
    ),
    axis=1,
).numpy()
gripper_closing_eef_euler = gripper_closing_eef_euler / np.pi * 180.0


gripper_closing_eef_pos = np.stack(gripper_closing_eef_pos_list, axis=0)

# fig, axs = plt.subplots(1, 2, figsize=(12, 5))
# axs[0].hist2d(gripper_closing_eef_pos[:, 1], gripper_closing_eef_pos[:, 0], bins=50)
# axs[0].set_title("Gripper Closing XY Position Heatmap")
# axs[1].hist(gripper_closing_eef_euler[:, 2], bins=50)
# axs[1].set_title("Gripper Closing Yaw Angle Histogram")
# plt.show()

plt.hist(gripper_closing_eef_euler[:, 1], bins=50)
plt.title("Gripper Closing Yaw Angle Histogram")
out_file_closing = os.path.join(f"{args.out_prefix}_gripper_closing_yaw_histogram.png")
plt.savefig(out_file_closing)
plt.show()
plt.clf()

print("Script finished.")
