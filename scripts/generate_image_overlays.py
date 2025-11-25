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

first_images = []

for in_file in in_files:

    with h5py.File(in_file, "r+") as hf:

        first_image = np.asarray(hf["obs"]["left_cam"]["frames"]["left"][0])
        first_images.append(first_image)

        # plt.imshow(first_image)
        # plt.axis('off')
        # plt.title(f"First Frame from {os.path.basename(in_file)}")
        # plt.show()

avg_image = np.mean(np.stack(first_images, axis=0), axis=0).astype(np.uint8)
plt.imshow(avg_image)
plt.axis("off")
plt.title("Average of First Frames")
plt.show()

plt.imsave(f"{args.out_prefix}_average_first_frame.png", avg_image)
print("Script finished.")
