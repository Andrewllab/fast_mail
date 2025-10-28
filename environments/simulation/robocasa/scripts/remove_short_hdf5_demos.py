"""
Automatically delete all demos from a hdf5-file with length less than threshold (e.g. for RoboCasa Microwave original dataset). In case contiguous indicies are desired please run rename_hdf5_groups.py
1. Some of the RoboCasa pre-recording human demos are not correct, e.g. only the first N frames were captured. 
2. Sometimes re-rendering using multiple processes leads to partially stored demos.
"""

import argparse
import h5py


def main():
    parser = argparse.ArgumentParser(description="Delete short demos from an HDF5 dataset.")
    parser.add_argument("path", type=str, help="Path to the HDF5 file.")
    parser.add_argument("--threshold", type=int, default=20,
                        help="Delete demos with less length than threshold (default: 20).")
    args = parser.parse_args()

    with h5py.File(args.path, "r+") as f:
        if "data" not in f:
            print("No 'data' group found in file.")
            return
        data_grp = f["data"]

        to_delete = []
        for demo_name, demo_grp in data_grp.items():
            if not demo_name.startswith("demo_"):
                continue
            if "states" not in demo_grp:
                continue
            n_steps = demo_grp["states"].shape[0]
            if n_steps <= args.threshold:
                to_delete.append(demo_name)

        if not to_delete:
            print(f"No demos found shorter than or equal to {args.threshold} steps.")
            return

        print(f"Deleting {len(to_delete)} demos: {to_delete}")
        for demo_name in to_delete:
            del data_grp[demo_name]

if __name__ == "__main__":
    main()
