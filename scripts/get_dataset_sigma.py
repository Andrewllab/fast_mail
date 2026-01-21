import glob
import os
import sys
from pathlib import Path

import h5py
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from utils.hdf5_utils import recursive_hdf5_to_dict
from utils.nested import unpack_nested_from_storage


def main():
    dataset_path = "/mnt/SmallSandwich/3d-sim2real_preprocessed/robocasa/kitchen_pnp_seg_pointclouds"
    dataset_files = glob.glob(f"{dataset_path}/*.hdf5")

    actions = []
    for dataset_path in dataset_files:
        with h5py.File(dataset_path, "r") as f:
            data = recursive_hdf5_to_dict(f)
            data = unpack_nested_from_storage(data)
            action = data["action"]
            actions.append(action.values())

    actions = torch.cat(actions, dim=0)  # (N, action_dim)
    sigma = torch.std(actions, dim=0)
    print("Dataset action sigma:", sigma)


if __name__ == "__main__":
    main()
