import argparse
import h5py

def main():
    parser = argparse.ArgumentParser(description="Delete a subgroup from an HDF5 file.")
    parser.add_argument(
        "path",
        type=str,
        help="Path to the HDF5 file",
    )
    parser.add_argument(
        "--group",
        type=str,
        required=True,
        help="Name of the subgroup to delete (e.g. 'demo_1', masks)",
    )
    args = parser.parse_args()

    with h5py.File(args.path, "r+") as f:
        if args.group in f:
            del f[args.group]
            print(f"Deleted group '{args.group}' and all its contents.")
        else:
            print(f"Group '{args.group}' not found. Top-level keys: {list(f.keys())}")

if __name__ == "__main__":
    main()
