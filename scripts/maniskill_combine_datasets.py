import os
import shutil
import glob
import argparse
import sys
import h5py
import numpy as np

BASE_DIRECTORY = "/s850x/demonstrations/maniskill"
MIN_STEPS = 10
STEPS_AFTER_SUCCESS = 10

def get_initial_step_count(file_path):
    """
    Safely opens an H5 file and returns the length of the 'actions' dataset.
    Returns 0 if the file is invalid, actions is missing, or actions is a scalar.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            if 'actions' in f:
                actions_ds = f['actions']
                if actions_ds.ndim > 0:
                    return actions_ds.shape[0]
    except Exception:
        pass
    return 0

def copy_and_truncate_group(source_group, dest_group, original_len_n, new_len_n):
    """
    Recursively copies groups and datasets from a source to a destination,
    truncating datasets that represent time-series data.
    """
    is_goal_related_group = source_group.name.startswith('/goal')

    for name, item in source_group.items():
        if isinstance(item, h5py.Dataset):
            should_slice = False
            
            if not is_goal_related_group and item.shape:
                if item.shape[0] == original_len_n:
                    data_to_write = item[:new_len_n]
                    should_slice = True
                elif item.shape[0] == original_len_n + 1:
                    data_to_write = item[:new_len_n + 1]
                    should_slice = True

            if should_slice:
                dest_group.create_dataset(name, data=data_to_write, dtype=item.dtype)
            else:
                source_group.copy(name, dest_group)

        elif isinstance(item, h5py.Group):
            new_group = dest_group.create_group(name)
            copy_and_truncate_group(item, new_group, original_len_n, new_len_n)


def process_and_truncate_h5(source_path, dest_path):
    """
    Opens an H5 file, determines if and where to truncate it based on the 'success'
    dataset, and saves a new truncated version.
    """
    try:
        with h5py.File(source_path, 'r') as f_in:
            if 'actions' not in f_in or 'success' not in f_in:
                print(f"    - Warning: Skipped '{os.path.basename(source_path)}' (missing essential datasets).")
                return 0

            actions_ds = f_in['actions']
            success_ds = f_in['success']

            if actions_ds.ndim == 0 or success_ds.ndim == 0:
                print(f"    - Warning: Skipped '{os.path.basename(source_path)}' (malformed scalar dataset found).")
                return 0

            success_arr = success_ds[:]
            original_num_steps = actions_ds.shape[0]
            success_indices = np.where(success_arr == True)[0]

            new_num_steps = original_num_steps
            if len(success_indices) > 0:
                first_success_idx = success_indices[0]
                potential_new_len = first_success_idx + STEPS_AFTER_SUCCESS
                new_num_steps = min(original_num_steps, potential_new_len)
            
            if new_num_steps < MIN_STEPS:
                print(f"    - Skipped '{os.path.basename(source_path)}': Final steps ({new_num_steps}) would be less than {MIN_STEPS}.")
                return 0

            if new_num_steps == original_num_steps:
                shutil.copy2(source_path, dest_path)
            else:
                with h5py.File(dest_path, 'w') as f_out:
                    copy_and_truncate_group(f_in, f_out, original_num_steps, new_num_steps)
            
            return new_num_steps

    except Exception as e:
        print(f"    - Error: Could not process '{source_path}'. Reason: {e}")
        return 0

def main():
    """
    Main function to parse arguments and combine datasets from the BASE_DIRECTORY.
    """
    parser = argparse.ArgumentParser(
        description=f"Combine and filter trajectory H5 files from subfolders within '{BASE_DIRECTORY}'.",
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

    total_files_processed = 0
    total_steps_copied = 0
    print("Starting file processing...")

    for folder_name in selected_folders:
        source_dir = os.path.join(root_dir, folder_name)
        traj_files = sorted(glob.glob(os.path.join(source_dir, 'traj_*.h5')))

        if not traj_files:
            print(f"  - No 'traj_*.h5' files found in '{folder_name}'. Skipping.")
            continue

        print(f"  - Processing '{folder_name}': Found {len(traj_files)} files.")
        
        processed_count = 0
        file_index = 0
        
        for source_path in traj_files:
            if amount_per_dataset is not None and processed_count >= amount_per_dataset:
                break
            
            original_steps = get_initial_step_count(source_path)

            if original_steps >= MIN_STEPS:
                dest_filename = f"traj_{folder_name}_{file_index:04d}.h5"
                dest_path = os.path.join(output_dir, dest_filename)
                
                original_relative_path = os.path.relpath(source_path, root_dir)

                final_steps = process_and_truncate_h5(source_path, dest_path)

                if final_steps > 0:
                    if final_steps < original_steps:
                        message = f"Processed and truncated '{original_relative_path}' -> '{os.path.join(output_folder_name, dest_filename)}' ({original_steps} -> {final_steps} steps)"
                    else:
                        message = f"Copied '{original_relative_path}' -> '{os.path.join(output_folder_name, dest_filename)}' ({final_steps} steps)"
                    print(f"    - {message}")
                    
                    processed_count += 1
                    total_files_processed += 1
                    total_steps_copied += final_steps
            elif original_steps > 0:
                print(f"    - Skipped '{os.path.relpath(source_path, root_dir)}': Only has {original_steps} steps (less than {MIN_STEPS}).")

            file_index += 1
        
        print(f"  - Finished processing '{folder_name}'. Processed {processed_count} files.")

    print(f"\nProcess complete. A total of {total_files_processed} files have been saved to the '{output_folder_name}' folder.")
    print(f"The combined dataset contains a total of {total_steps_copied} steps.")

if __name__ == "__main__":
    if not os.path.isdir(BASE_DIRECTORY):
        print(f"Error: The configured BASE_DIRECTORY does not exist or is not a directory.")
        print(f"Path: '{BASE_DIRECTORY}'")
        sys.exit(1)
    main()