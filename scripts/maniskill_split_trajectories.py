import argparse
from pathlib import Path
from tensordict import TensorDict
from tqdm import tqdm


def split_trajectories(input_file: Path, output_dir: Path):
    """
    Reads a single HDF5 file containing multiple trajectories and saves each
    trajectory into its own separate HDF5 file.

    Args:
        input_file (Path): The path to the source H5 file.
        output_dir (Path): The directory where the new H5 files will be saved.
    """
    if not input_file.is_file():
        print(f"File not found {input_file}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    source_h5 = TensorDict.from_h5(str(input_file), mode="r")

    traj_keys = sorted([key for key in source_h5.keys() if key.startswith("traj")])

    if not traj_keys:
        print(f"No keys starting with 'traj'")
        return

    for i, key in enumerate(tqdm(traj_keys, desc="Splitting Trajectories")):
        trajectory_td = source_h5[key]

        output_path = output_dir / f"traj_{i:04d}.h5"

        trajectory_td.to_h5(output_path)

    print(f" Done. {len(traj_keys)} files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a single HDF5 file with multiple trajectories into many HDF5 files with one trajectory each."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the large input HDF5 file."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the directory where the split files will be saved.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    split_trajectories(input_path, output_path)
