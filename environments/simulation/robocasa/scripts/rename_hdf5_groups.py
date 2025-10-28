"""
Automatically rename all demos in hdf5 file to have contiguous indicies, e.g. after removing intermediate correpted demos. 
"""
import argparse
import h5py


def main():
    parser = argparse.ArgumentParser(description="Make demo counters contiguous again in an HDF5 file.")
    parser.add_argument("path", type=str, help="Path to the HDF5 file.")
    parser.add_argument("--root-group", type=str, default="data",
                        help="Root group containing demo_* (default: data).")
    args = parser.parse_args()

    with h5py.File(args.path, "r+") as f:
        if args.root_group not in f:
            print(f"Group '{args.root_group}' not found in file.")
            return
        root = f[args.root_group]

        # Get existing demos sorted by numeric suffix
        demo_names = [name for name in root.keys() if name.startswith("demo_")]
        demo_names_sorted = sorted(demo_names, key=lambda n: int(n.split("_")[1]))

        print(f"Found {len(demo_names_sorted)} demos.")
        new_index = 1
        for old_name in demo_names_sorted:
            new_name = f"demo_{new_index}"
            if old_name == new_name:
                # already contiguous
                new_index += 1
                continue
            print(f"Renaming {old_name} -> {new_name}")
            # copy data
            root.copy(old_name, new_name)
            # delete old
            del root[old_name]
            new_index += 1

        print(f"Now {len(root.keys())} demos, contiguous from demo_1 to demo_{new_index-1}")

if __name__ == "__main__":
    main()
