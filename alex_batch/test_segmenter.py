"""Visualize batched SAM3 prompt and click masks on initial camera frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from alex.simulate_static import HDF5FrameReader
if __package__:
    from .track_online_pipe import SAM3Segmenter
else:
    from track_online_pipe import SAM3Segmenter


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="/home/david/projects_wr/2026_05_18-12_49_11.h5")
    parser.add_argument("--prompt", default="red cup")
    parser.add_argument("--clicks-json", help="camera -> object -> {points, labels} JSON")
    parser.add_argument("--output", default="alex_batch/outputs/segmenter_test.png")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(arguments)
    camera_paths = {"left_cam": "obs/left_cam/frames/left", "right_cam": "obs/right_cam/frames/left"}
    with HDF5FrameReader(args.path, camera_paths=camera_paths, stop=1) as reader:
        bundle = reader.next_frame()
    if bundle is None:
        raise RuntimeError("Episode contains no frames")
    images = bundle.images
    clicks = json.loads(args.clicks_json) if args.clicks_json else {
        camera: {"object": {"points": [[frame.shape[1] / 2, frame.shape[0] / 2]], "labels": [1]}}
        for camera, frame in images.items()
    }
    prompt_selection = {camera: [{"name": "prompt", "text": args.prompt}] for camera in images}
    click_selection = {
        camera: [{"name": name, "points": value["points"], "labels": value.get("labels", [1] * len(value["points"]))}
                 for name, value in clicks[camera].items()]
        for camera in images
    }
    segmenter = SAM3Segmenter(list(images), device=args.device)
    prompt_masks = segmenter.segment(images, prompt_selection)
    segmenter._models.pop("text", None)
    click_masks = segmenter.segment(images, click_selection)
    figure, axes = plt.subplots(3, len(images), squeeze=False, figsize=(5 * len(images), 11))
    for column, camera in enumerate(images):
        overlays = [images[camera], images[camera].copy(), images[camera].copy()]
        prompt_mask = prompt_masks[camera]["prompt"]
        overlays[1][prompt_mask] = (0.7 * overlays[1][prompt_mask] + np.array([0, 220, 70]) * 0.3).astype(np.uint8)
        click_mask = next(iter(click_masks[camera].values()))
        overlays[2][click_mask] = (0.7 * overlays[2][click_mask] + np.array([0, 220, 70]) * 0.3).astype(np.uint8)
        titles = [f"{camera} | original", f"{camera} | prompt: {args.prompt}", f"{camera} | click"]
        for row in range(3):
            axes[row, column].imshow(overlays[row])
            axes[row, column].set_title(titles[row])
            axes[row, column].axis("off")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved batched prompt/click comparison for {len(images)} cameras to {output}")


if __name__ == "__main__":
    main()

