import argparse
import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import imageio
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


def darken_outside_mask_batch(
    images: np.ndarray, masks: np.ndarray, darken_factor: float = 0.4
) -> np.ndarray:
    """
    Vectorized version that operates on a full batch of frames.
    images: uint8 array of shape (N, H, W, 3)
    masks:  bool array of shape (N, H, W), True = keep highlighted
    """
    out = images.astype(np.float32)
    out[~masks] *= darken_factor
    return np.clip(out, 0, 255).astype(np.uint8)


def save_image(args):
    path, img = args
    imageio.imwrite(path, img)


if __name__ == "__main__":

    in_folder = (
        "/mnt/SmallSandwich/pakt_preprepreprocessed/real/stack_20260116_pointclouds"
    )
    out_folder = "/mnt/SmallSandwich/visualization/stack_20260116_pointclouds"
    cam_keys = ["front_left_cam", "front_right_cam"]
    seg_keys = ["tool_segmentation", "target_segmentation"]

    os.makedirs(out_folder, exist_ok=True)

    for in_file in glob.glob(os.path.join(in_folder, "**/*.hdf5"), recursive=True):
        with h5py.File(in_file, "r") as hf:  # read-only, no need for "r+"
            num_frames = hf["obs"][cam_keys[0]]["left"].shape[0]
            rel_path = os.path.relpath(in_file, in_folder).split(".")[0]
            rel_out_folder = os.path.join(out_folder, rel_path)
            os.makedirs(rel_out_folder, exist_ok=True)

            print(f"Loading {in_file} into memory...")

            # --- Read all data upfront in one pass ---
            cam_data = {}
            for cam_key in cam_keys:
                rgb_all = np.asarray(
                    hf["obs"][cam_key]["left"][:], dtype=np.uint8
                )  # (N, H, W, 3)
                seg_all = {
                    seg_key: np.asarray(hf["obs"][cam_key][seg_key][:], dtype=bool)
                    for seg_key in seg_keys
                }

                # Combined segmentation mask: (N, H, W)
                comb_mask = np.zeros(rgb_all.shape[:3], dtype=bool)  # (N, H, W)
                for seg_key in seg_keys:
                    comb_mask |= seg_all[seg_key]

                # Darken entire batch at once
                rgb_darkened = darken_outside_mask_batch(rgb_all, comb_mask)

                cam_data[cam_key] = {"rgb": rgb_darkened, "seg": seg_all}

            # --- Assemble output frames ---
            print(f"Assembling {num_frames} frames...")
            output_frames = []
            for i in range(num_frames):
                row_images = []
                for cam_key in cam_keys:
                    rgb = cam_data[cam_key]["rgb"][i]  # (H, W, 3)
                    local_out = rgb
                    for seg_key in seg_keys:
                        seg_frame = cam_data[cam_key]["seg"][seg_key][i]  # (H, W) bool
                        seg_vis = (seg_frame[..., None].astype(np.uint8) * 255).repeat(
                            3, axis=-1
                        )
                        local_out = np.concatenate([local_out, seg_vis], axis=1)
                    row_images.append(local_out)
                output_frames.append(np.concatenate(row_images, axis=0))

            # --- Save in parallel ---
            print(f"Saving {num_frames} frames...")
            paths = [
                os.path.join(rel_out_folder, f"{i:04d}.png") for i in range(num_frames)
            ]
            with ThreadPoolExecutor(max_workers=8) as executor:
                executor.map(save_image, zip(paths, output_frames))

        print(f"Done with {in_file}.")
