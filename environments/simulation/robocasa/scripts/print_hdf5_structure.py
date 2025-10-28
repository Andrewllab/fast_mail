import argparse
import h5py

def print_hdf5_structure(filename):
    """Print all groups and datasets in an HDF5 file."""
    def print_attrs(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"{name}/ (group)")
    with h5py.File(filename, "r") as f:
        f.visititems(print_attrs)

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print HDF5-file structure with all sub-groups."
    )
    parser.add_argument("path", type=str, help="Path to the HDF5 file.")
    args = parser.parse_args()

    print_hdf5_structure(args.path)
