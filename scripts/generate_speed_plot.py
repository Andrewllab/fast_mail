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
    print(f"Processing file: {in_file}")

    with h5py.File(in_file, "r+") as hf:

        gripper_pos = np.asarray(hf["obs"]["proprioception"]["gripper_pos"])
        eef_pos = np.asarray(hf["obs"]["proprioception"]["eef_pos"])
        gripper_vel = np.diff(gripper_pos, axis=0)
        
        joint_vel = np.diff(hf["obs"]["proprioception"]["joint_pos"], axis=0)
        num_joints = joint_vel.shape[1]
        
        fig, axs = plt.subplots(num_joints + 1, 1, figsize=(8, 2 * (num_joints + 1)), sharex=True)

        for i in range(num_joints):
            axs[i].plot(joint_vel[:, i], label=f"Joint {i+1} Velocity")
            axs[i].set_ylabel("Velocity (rad/s)")
            axs[i].legend()
            axs[i].grid(True)
        axs[-1].plot(gripper_vel, label="Gripper Velocity", color='orange')
        axs[-1].set_ylabel("Velocity (m/s)")
        axs[-1].set_xlabel("Time Step")
        axs[-1].legend()
        axs[-1].grid(True)
        plt.suptitle("Joint and Gripper Velocities Over Time")
        plt.show()

        joint_vel_sum = np.linalg.norm(joint_vel, axis=1)
        fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        
        axs[0].plot(joint_vel_sum, label="Sum of Joint Velocities")
        axs[0].set_ylabel("Sum of Velocities (rad/s)")
        axs[0].legend()
        axs[0].grid(True)
        axs[1].plot(gripper_vel, label="Gripper Velocity", color='orange')
        axs[1].set_ylabel("Velocity (m/s)")
        axs[1].set_xlabel("Time Step")
        axs[1].legend()
        axs[1].grid(True)
        plt.suptitle("Sum of Joint Velocities and Gripper Velocity Over Time")
        plt.show()


        
print("Script finished.")