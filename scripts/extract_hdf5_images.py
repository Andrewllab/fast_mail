import argparse
import glob
import os
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.rendering import depth_to_renderable

in_file = "/mnt/SmallSandwich/2026_01_20-17_52_25.h5"
out_folder = "/mnt/SmallSandwich/visualization"
os.makedirs(out_folder, exist_ok=True)

with h5py.File(in_file, "r+") as hf:
    # for cam in ["left_cam", "right_cam", "gripper_cam"]:
    # cam_out_folder = os.path.join(args.out_folder, cam)
    # os.makedirs(cam_out_folder, exist_ok=True)

    # depth_key = "depth"
    # if cam != "gripper_cam":
    #     rgb_key = "left"
    # else:
    #     rgb_key = "rgb"

    # rgb_images = np.asarray(hf["obs"][cam]["frames"][rgb_key])
    # depth_images = np.asarray(hf["obs"][cam]["frames"][depth_key])
    # rgb_images = np.asarray(hf["obs"]["left_cam"]["left"])
    rgb_images = np.asarray(hf["obs"]["left_cam"]["frames"]["left"])

    # depth_images = np.asarray(
    #     hf["data"]["demo_2"]["obs"]["robot0_agentview_left_depth"]
    # )

    rgb_folder = os.path.join(out_folder, "rgb")
    os.makedirs(rgb_folder, exist_ok=True)

    for i in range(rgb_images.shape[0]):
        rgb_image = rgb_images[i]
        rgb_filename = os.path.join(rgb_folder, f"{i:04d}.png")
        plt.imsave(rgb_filename, rgb_image)

    # depth_folder = os.path.join(args.out_folder, "depth")
    # os.makedirs(depth_folder, exist_ok=True)

    # for i in range(0, depth_images.shape[0], 10):
    #     depth_image = depth_images[i]
    #     depth_filename = os.path.join(depth_folder, f"{i:04d}.png")

    #     depth_renderable = depth_to_renderable(
    #         depth_image, depth_min=0.0, depth_max=1, colormap_name="magma"
    #     )
    #     plt.imsave(depth_filename, depth_renderable)
