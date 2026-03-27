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



def main():
    dataset_path = "/mnt/LargeSandwich/PAKT/demonstrations/robocasa/single_stage/**/*.hdf5"

    dataset_files = glob.glob(dataset_path, recursive=True)

    len_list = []

    for dataset_file in dataset_files:

        with h5py.File(dataset_file, "r") as f:
            for demo_key in f["data"].keys():
                data = f[f"data/{demo_key}"]

                for seg_key in data["segmentation_ids"].keys():
                    seg_ids = data[f"segmentation_ids/{seg_key}"][:]
                    len_list.append(len(seg_ids))
                pass
    
    print(f"Max segmentation ids length: {max(len_list)}")
    print(f"Mean segmentation ids length: {sum(len_list) / len(len_list)}")
    print(f"Min segmentation ids length: {min(len_list)}")
    

if __name__ == "__main__":
    main()
