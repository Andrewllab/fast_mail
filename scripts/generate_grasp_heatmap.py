import sys
import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

parser = argparse.ArgumentParser(description="Inject a pre-processed goal embedding into an HDF5 file.")
parser.add_argument("--folderpath", type=str, required=True, help="Path to the folder containing all .h5 files")
parser.add_argument("--out-prefix", type=str, required=True, help="Prefix for the name of the generated heatmaps")
args = parser.parse_args()

in_files = glob.glob(os.path.join(args.folderpath, "**","*.h5"), recursive=True)

border_value = 0.055

gripper_closing_eef_pos_list = []
gripper_opening_eef_pos_list = []

max_workspace = np.array([-np.inf, -np.inf, -np.inf])
min_workspace = np.array([np.inf, np.inf, np.inf])

for in_file in in_files:

    with h5py.File(in_file, "r+") as hf:

        gripper_pos = np.asarray(hf["action"]["gripper_pos"])
        eef_pos = np.asarray(hf["obs"]["proprioception"]["eef_pos"])
        
        gripper_closing = np.all([gripper_pos[:-1] > border_value, gripper_pos[1:] <= border_value], axis=0)
        gripper_opening = np.all([gripper_pos[:-1] <= border_value, gripper_pos[1:] > border_value], axis=0)
        
        gripper_closing_indices = np.where(gripper_closing)[0] + 1
        gripper_opening_indices = np.where(gripper_opening)[0] + 1

        gripper_closing_eef_pos = eef_pos[gripper_closing_indices]
        gripper_opening_eef_pos = eef_pos[gripper_opening_indices]
        gripper_closing_eef_pos_list.append(gripper_closing_eef_pos)
        gripper_opening_eef_pos_list.append(gripper_opening_eef_pos)

        max_workspace = np.maximum(max_workspace, np.max(eef_pos, axis=0))
        min_workspace = np.minimum(min_workspace, np.min(eef_pos, axis=0))

        gripper_pos_diff = np.diff(gripper_pos, axis=0)
        plt.plot(gripper_pos_diff)
        plt.plot(gripper_pos)
        plt.plot(gripper_closing)
        plt.show()

gripper_closing_eef_pos = np.concatenate(gripper_closing_eef_pos_list, axis=0)

plt.hist2d(gripper_closing_eef_pos[:,1], gripper_closing_eef_pos[:,0], bins=50, range=[[min_workspace[1], max_workspace[1]], [min_workspace[0], max_workspace[0]]])
plt.title("Gripper Closing Heatmap")
out_file_closing = os.path.join(f"{args.out_prefix}_gripper_closing_heatmap.png")
plt.savefig(out_file_closing)
plt.clf()

gripper_opening_eef_pos = np.concatenate(gripper_opening_eef_pos_list, axis=0)
plt.hist2d(gripper_opening_eef_pos[:,1], gripper_opening_eef_pos[:,0], bins=50, range=[[min_workspace[1], max_workspace[1]], [min_workspace[0], max_workspace[0]]])
plt.title("Gripper Opening Heatmap")
out_file_opening = os.path.join(f"{args.out_prefix}_gripper_opening_heatmap.png")
plt.savefig(out_file_opening)
plt.clf()


        


        
print("Script finished.")