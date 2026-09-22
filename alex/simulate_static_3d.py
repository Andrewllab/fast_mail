"""Replay an RGB-D HDF5 episode through the online 3D tracking pipeline.

The RGB stream follows ``simulate_static.py``.  At each completed CoTracker
window (frames 15, 23, 31, ...) the final 2D prediction is lifted using the
corresponding depth image, intrinsics, and camera-to-world calibration.  Use
``--synthetic-depth`` only for recordings without depth (such as the example
cup recording); those coordinates are not metric 3D measurements.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch

# Make ``python alex/simulate_static_3d.py`` behave like the existing script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alex.simulate_static import HDF5FrameReader, _merge
from alex_3d.selection import DirectPointSegmenter, collect_direct_points
from alex_3d.track_online_pipe import CameraCalibration


def _config(arguments):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--output-dir", type=Path, default=Path("alex/outputs/cups_3d"))
    parser.add_argument("--depth-path", action="append", default=[], metavar="CAMERA=HDF5_PATH")
    parser.add_argument("--calibration-json", type=Path)
    parser.add_argument("--synthetic-depth", action="store_true")
    parser.add_argument("--depth-scale", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args(arguments)

    base = {
        "source": {
            "path": "/home/david/projects_wr/2026_05_18-12_49_11.h5",
            "camera_paths": {
                "left_cam": "obs/left_cam/frames/left",
                "right_cam": "obs/right_cam/frames/left",
            },
            "start": 0, "stop": None, "stride": 1,
            "resize_wh": None, "color_order": "RGB", "float_range": "0_255",
        },
        "device": "auto", "cpu_threads": 4,
        "selection": "mask_click",
        "selections": None,
        "segmenter": {"model_id": "facebook/sam3", "cache_dir": str(Path(__file__).parent / ".cache/huggingface"),
                       "local_files_only": False, "threshold": 0.5},
        "tracker": {"grid_spacing": 16, "max_points_per_object": 128, "support_grid": True,
                    "hub_repo": "facebookresearch/co-tracker", "hub_source": "github"},
        "hub_dir": str(Path(__file__).parent / ".cache/torch/hub"),
        "objects": ["object"],
    }
    if args.config:
        _merge(base, json.loads(args.config.read_text()))
    for override in args.set:
        if "=" not in override:
            parser.error("--set requires dotted.key=value")
        dotted, raw = override.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = base
        parts = dotted.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                parser.error(f"Unknown dictionary path: {dotted}")
            node = node[part]
        if parts[-1] not in node:
            parser.error(f"Unknown configuration key: {dotted}")
        node[parts[-1]] = value
    depth_paths = {}
    for item in args.depth_path:
        if "=" not in item:
            parser.error("--depth-path requires CAMERA=HDF5_PATH")
        camera, path = item.split("=", 1)
        depth_paths[camera] = path
    return args, base, depth_paths


def _calibrations(handle, cameras, calibration_json, depth_scale):
    overrides = json.loads(calibration_json.read_text()) if calibration_json else {}
    result = {}
    for camera in cameras:
        if camera in overrides:
            result[camera] = CameraCalibration.from_mapping(overrides[camera])
            continue
        intrinsics = handle[f"obs/{camera}/meta/intrinsics"][()]
        result[camera] = CameraCalibration(intrinsics, np.eye(4), depth_scale=depth_scale)
        warnings.warn(
            f"No world extrinsics supplied for {camera}; using identity. "
            "Provide --calibration-json for metric world coordinates.",
            stacklevel=2,
        )
    return result


def run(arguments=None, batched=False):
    args, config, depth_paths = _config(arguments)
    if args.depth_scale <= 0:
        raise ValueError("--depth-scale must be positive")
    if config["selection"] not in ("mask_click", "keypoints"):
        raise ValueError("selection must be 'mask_click' or 'keypoints'")
    if not config["objects"] or len(set(config["objects"])) != len(config["objects"]):
        raise ValueError("objects must contain unique names")
    torch.set_num_threads(int(config["cpu_threads"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if batched:
        from alex_batch_3d.track_online_pipe import OnlineKeypointTracker, OnlineTrackingPipeline, SAM3Segmenter
    else:
        from alex_3d.track_online_pipe import OnlineKeypointTracker, OnlineTrackingPipeline, SAM3Segmenter

    torch.hub.set_dir(config["hub_dir"])
    with HDF5FrameReader(**config["source"]) as source, h5py.File(config["source"]["path"], "r") as handle:
        first = source.next_frame()
        if first is None:
            raise RuntimeError("The selected source interval contains no frames")
        cameras = list(first.images)
        if args.synthetic_depth:
            depth_paths = {}
        elif not depth_paths:
            depth_paths = {camera: f"obs/{camera}/frames/depth" for camera in cameras}
        for camera in cameras:
            if not args.synthetic_depth and depth_paths[camera] not in handle:
                raise KeyError(
                    f"Depth dataset {depth_paths[camera]!r} is missing for {camera}; "
                    "use --depth-path or --synthetic-depth."
                )

        selections = config["selections"]
        if selections is None:
            selections = (
                collect_direct_points(first.images, config["objects"])
                if config["selection"] == "keypoints"
                else SAM3Segmenter.collect_clicks(first.images, config["objects"])
            )
        calibrations = _calibrations(handle, cameras, args.calibration_json, args.depth_scale)
        segmenter = (
            DirectPointSegmenter(cameras) if config["selection"] == "keypoints"
            else SAM3Segmenter(cameras, device=config["device"], **config["segmenter"])
        )
        tracker = OnlineKeypointTracker(cameras, device=config["device"], **config["tracker"])
        pipeline = OnlineTrackingPipeline(segmenter, tracker, calibrations, keep_history=True)

        if config["selection"] == "keypoints":
            pipeline.initialize_points(first.images, selections)
        else:
            pipeline.initialize(first.images, selections)
        bundle = first
        while bundle is not None:
            if bundle.index > 0:
                depth_images = {}
                for camera in cameras:
                    if args.synthetic_depth:
                        depth_images[camera] = np.ones(bundle.images[camera].shape[:2], dtype=np.float32)
                    else:
                        depth = handle[depth_paths[camera]][bundle.source_index][()]
                        if depth.ndim == 3:
                            depth = depth[..., 0]
                        depth_images[camera] = np.asarray(depth)
                pipeline.push(bundle.images, depth_images)
            bundle = source.next_frame()
        pipeline.finish()

    pipeline.export(args.output_dir, fps=args.fps, export_3d=True)
    print(f"Wrote 2D and 3D outputs to {args.output_dir}")


def main(arguments=None):
    args, _, _ = _config(arguments)
    run(arguments, batched=args.output_dir.name.startswith("batch"))


if __name__ == "__main__":
    main()
