import os
import shutil
import glob
import argparse
import sys

BASE_DIRECTORY = "/s850x/demonstrations/maniskill"

def main():
    """
    Main function to parse arguments and combine datasets from the BASE_DIRECTORY.
    """
    parser = argparse.ArgumentParser(
        description=f"Combine trajectory H5 files from subfolders within '{BASE_DIRECTORY}'.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help="Name for the new output subfolder where combined files will be stored."
    )
    parser.add_argument(
        '-d', '--datasets',
        type=str,
        nargs='+',
        required=True,
        help="A space-separated list of the dataset subfolder names to combine."
    )
    parser.add_argument(
        '-n', '--amount',
        type=int,
        default=None,
        help="Optional: Maximum number of trajectories to select from EACH dataset."
    )

    args = parser.parse_args()

    root_dir = BASE_DIRECTORY
    output_folder_name = args.output
    selected_folders = args.datasets
    amount_per_dataset = args.amount

    for folder in selected_folders:
        path = os.path.join(root_dir, folder)
        if not os.path.isdir(path):
            print(f"Error: The specified dataset folder does not exist: '{path}'")
            return

    output_dir = os.path.join(root_dir, output_folder_name)
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Base Directory: '{root_dir}'")
        print(f"Output directory ready: '{output_dir}'")
        print(f"Selected datasets: {', '.join(selected_folders)}")
        if amount_per_dataset:
            print(f"Amount to select per dataset: {amount_per_dataset}\n")
    except OSError as e:
        print(f"Error creating directory '{output_dir}': {e}")
        return

    total_files_copied = 0
    print("Starting file copy process...")

    for folder_name in selected_folders:
        source_dir = os.path.join(root_dir, folder_name)
        traj_files = sorted(glob.glob(os.path.join(source_dir, 'traj_*.h5')))

        if not traj_files:
            print(f"  - No 'traj_*.h5' files found in '{folder_name}'. Skipping.")
            continue

        if amount_per_dataset is not None:
            if amount_per_dataset <= 0:
                print("Error: --amount must be a positive number.")
                return
            files_to_copy = traj_files[:amount_per_dataset]
        else:
            files_to_copy = traj_files

        print(f"  - Processing '{folder_name}': Found {len(traj_files)} files, will copy {len(files_to_copy)}.")

        per_dataset_counter = 0

        for source_path in files_to_copy:
            dest_filename = f"traj_{folder_name}_{per_dataset_counter:04d}.h5"
            dest_path = os.path.join(output_dir, dest_filename)
            
            original_relative_path = os.path.relpath(source_path, root_dir)

            try:
                shutil.copy2(source_path, dest_path)
                print(f"    - Copied '{original_relative_path}' -> '{os.path.join(output_folder_name, dest_filename)}'")
                per_dataset_counter += 1
                total_files_copied += 1
            except Exception as e:
                print(f"    - Error: Could not copy '{source_path}'. Reason: {e}")

    print(f"\nProcess complete. A total of {total_files_copied} files have been combined into the '{output_folder_name}' folder.")


if __name__ == "__main__":
    if not os.path.isdir(BASE_DIRECTORY):
        print(f"Error: The configured BASE_DIRECTORY does not exist or is not a directory.")
        print(f"Path: '{BASE_DIRECTORY}'")
        sys.exit(1)
    main()