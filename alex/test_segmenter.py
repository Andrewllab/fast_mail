"""Test SAM3 prompt and click segmentation on the first HDF5 frame.

The output PNG has rows for original images, text-prompt masks, and click masks;
columns correspond to camera views. It requires approved access to
``facebook/sam3`` and a working CUDA/CPU Transformers installation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from .simulate_static import HDF5FrameReader
    from .track_online_pipe import SAM3Segmenter
else:
    from simulate_static import HDF5FrameReader
    from track_online_pipe import SAM3Segmenter


DEFAULT_CAMERA_PATHS = {
    "left_cam": "obs/left_cam/frames/left",
    "right_cam": "obs/right_cam/frames/left",
}


def _parse_json(value: str, description: str):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {description} JSON: {error}") from error
    return parsed


def _normalise_clicks(clicks, images):
    if clicks is None:
        # A deterministic fallback keeps the script runnable without a GUI. For
        # useful segmentation, provide --clicks-json with object-specific clicks.
        return {
            camera: {
                "object": {
                    "points": [[frame.shape[1] / 2, frame.shape[0] / 2]],
                    "labels": [1],
                }
            }
            for camera, frame in images.items()
        }
    if not isinstance(clicks, dict):
        raise ValueError("clicks JSON must be a camera -> object -> click mapping")
    normalised = {}
    for camera, frame in images.items():
        camera_clicks = clicks.get(camera)
        if not isinstance(camera_clicks, dict) or not camera_clicks:
            raise ValueError(f"Missing nonempty click objects for camera {camera}")
        normalised[camera] = {}
        for name, value in camera_clicks.items():
            if isinstance(value, dict):
                points = value.get("points")
                labels = value.get("labels", [1] * len(points or []))
            else:
                points, labels = value, None
            if points is None:
                raise ValueError(f"{camera}/{name}: expected points")
            normalised[camera][name] = {
                "points": points,
                "labels": labels or [1] * len(points),
            }
    return normalised


def _make_selection(images, prompt, click_objects):
    prompts = {
        camera: [{"name": "prompt", "text": prompt}]
        for camera in images
    }
    clicks = {
        camera: [
            {"name": name, "points": value["points"], "labels": value["labels"]}
            for name, value in objects.items()
        ]
        for camera, objects in click_objects.items()
    }
    return prompts, clicks


def _overlay(image, mask, title):
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match image {image.shape[:2]}")
    result = image.astype(np.float32).copy()
    result[mask] = 0.70 * result[mask] + 0.30 * np.array([40, 220, 70])
    return result.astype(np.uint8), title


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default="/home/david/projects_wr/2026_05_18-12_49_11.h5",
        help="Input HDF5 episode",
    )
    parser.add_argument(
        "--camera-paths-json",
        default=json.dumps(DEFAULT_CAMERA_PATHS),
        help="JSON mapping camera names to HDF5 image dataset paths",
    )
    parser.add_argument("--prompt", default="red cup")
    parser.add_argument(
        "--clicks-json",
        help=(
            "JSON mapping camera -> object -> {points:[[x,y],...], labels:[1,0,...]}. "
            "If omitted, one foreground click at each image center is used."
        ),
    )
    parser.add_argument("--output", default="alex/outputs/segmenter_test.png")
    parser.add_argument("--masks-output", help="Optional .npz path for raw prompt/click masks")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model-id", default="facebook/sam3")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    camera_paths = _parse_json(args.camera_paths_json, "camera paths")
    if not isinstance(camera_paths, dict) or not camera_paths:
        raise ValueError("camera paths must be a nonempty JSON object")
    with HDF5FrameReader(args.path, camera_paths=camera_paths, stop=1) as reader:
        bundle = reader.next_frame()
    if bundle is None:
        raise RuntimeError("The input episode has no frame at its selected start")

    images = bundle.images
    click_objects = _normalise_clicks(
        _parse_json(args.clicks_json, "clicks") if args.clicks_json else None,
        images,
    )
    prompt_selections, click_selections = _make_selection(
        images, args.prompt, click_objects
    )
    segmenter = SAM3Segmenter(
        list(images),
        device=args.device,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        threshold=args.threshold,
    )

    prompt_masks = segmenter.segment(images, prompt_selections)
    segmenter._models.pop("text", None)
    click_masks = segmenter.segment(images, click_selections)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        3, len(images), squeeze=False,
        figsize=(5 * len(images), 11), constrained_layout=True,
    )
    for column, camera in enumerate(images):
        prompt_mask = prompt_masks[camera]["prompt"]
        click_mask = next(iter(click_masks[camera].values()))
        prompt_overlay, prompt_title = _overlay(
            images[camera], prompt_mask, f"{camera} | prompt: {args.prompt}"
        )
        click_name = next(iter(click_masks[camera]))
        click_overlay, click_title = _overlay(
            images[camera], click_mask, f"{camera} | click: {click_name}"
        )
        rows = (
            (images[camera], f"{camera} | original"),
            (prompt_overlay, prompt_title),
            (click_overlay, click_title),
        )
        for row, (overlay, title) in enumerate(rows):
            axes[row, column].imshow(overlay)
            axes[row, column].set_title(title)
            axes[row, column].axis("off")
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Prompt mask", fontsize=12)
    axes[2, 0].set_ylabel("Click mask", fontsize=12)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)

    if args.masks_output:
        masks_path = Path(args.masks_output)
        masks_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            masks_path,
            prompt_masks=np.stack(
                [prompt_masks[camera]["prompt"] for camera in images]
            ),
            click_masks=np.stack(
                [next(iter(click_masks[camera].values())) for camera in images]
            ),
            cameras=np.array(list(images)),
        )
    print(f"Saved prompt/click comparison for {len(images)} cameras to {output}")
    if args.masks_output:
        print(f"Saved raw masks to {args.masks_output}")


if __name__ == "__main__":
    main()
