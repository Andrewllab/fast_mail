"""Replay an HDF5 episode through batched SAM3/CoTracker3."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alex.simulate_static import HDF5FrameReader
from alex.track_online_pipe import _safe_name
from alex_3d.selection import DirectPointSegmenter, collect_direct_points
if __package__:
    from .track_online_pipe import OnlineKeypointTracker, OnlineTrackingPipeline, SAM3Segmenter
else:
    from track_online_pipe import OnlineKeypointTracker, OnlineTrackingPipeline, SAM3Segmenter


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = {
    "source": {"path": "/home/david/projects_wr/2026_05_18-12_49_11.h5", "camera_paths": None,
               "start": 0, "stop": None, "stride": 1, "resize_wh": None,
               "color_order": "RGB", "float_range": "0_255"},
    "device": "auto", "cpu_threads": 4, "objects": ["object"],
    "selection": "mask_click", "selections": None,
    "segmenter": {"model_id": "facebook/sam3", "cache_dir": str(HERE / ".cache" / "huggingface"),
                   "local_files_only": False, "threshold": 0.5},
    "tracker": {"grid_spacing": 16, "max_points_per_object": 128, "support_grid": True,
                "hub_repo": "facebookresearch/co-tracker", "hub_source": "github"},
    "hub_dir": str(HERE / ".cache" / "torch" / "hub"), "output_dir": str(HERE / "outputs"),
    "fps": 15.0, "save_videos": True, "trail_length": 20,
    "show_initial": False, "show_live": False, "play_final": False,
}


def _merge(base, update):
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(base[key], dict) and isinstance(value, dict):
            _merge(base[key], value)
        else:
            base[key] = value


def read_config(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(arguments)
    config = copy.deepcopy(DEFAULT_CONFIG)
    if args.config:
        _merge(config, json.loads(args.config.read_text()))
    for override in args.set:
        if "=" not in override:
            parser.error("--set requires dotted.key=value")
        dotted, raw = override.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                parser.error(f"Unknown dictionary path: {dotted}")
            node = node[part]
        if parts[-1] not in node:
            parser.error(f"Unknown configuration key: {dotted}")
        node[parts[-1]] = value
    return config


def _write_videos(output_dir, camera_names, frames, results, fps, trail_length):
    paths = {}
    for camera in camera_names:
        camera_results = results[camera]
        if not camera_results:
            continue
        height, width = frames[camera][0].shape[:2]
        path = Path(output_dir) / f"{_safe_name(camera)}_sam3_cotracker.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (width + width % 2, height + height % 2))
        if not writer.isOpened():
            raise OSError(f"Video encoder could not open {path}")
        try:
            for result_index, result in enumerate(camera_results):
                frame = frames[camera][result.frame_index].copy()
                if trail_length > 0:
                    first_segment = max(0, result_index - trail_length)
                    for segment_index in range(first_segment, result_index):
                        previous = camera_results[segment_index]
                        following = camera_results[segment_index + 1]
                        for point_index, (start_visible, end_visible) in enumerate(
                                zip(previous.visible, following.visible)):
                            if start_visible and end_visible:
                                cv2.line(frame,
                                         tuple(np.round(previous.xy[point_index]).astype(int)),
                                         tuple(np.round(following.xy[point_index]).astype(int)),
                                         (255, 180, 0), 1, cv2.LINE_AA)
                overlay = OnlineKeypointTracker.visualize({camera: frame}, {camera: result})[camera]
                overlay = cv2.copyMakeBorder(overlay, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        paths[camera] = str(path)
    return paths


def main(arguments=None):
    config = read_config(arguments)
    if config["selection"] not in ("mask_click", "keypoints"):
        raise ValueError("selection must be 'mask_click' or 'keypoints'")
    if not config["objects"] or len(set(config["objects"])) != len(config["objects"]):
        raise ValueError("objects must contain unique names")
    torch.set_num_threads(int(config["cpu_threads"]))
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with HDF5FrameReader(**config["source"]) as source:
        first = source.next_frame()
        if first is None:
            raise RuntimeError("The selected source interval contains no frames")
        camera_names = list(first.images)
        selections = config["selections"]
        if selections is None:
            selections = (
                collect_direct_points(first.images, config["objects"])
                if config["selection"] == "keypoints"
                else SAM3Segmenter.collect_clicks(first.images, config["objects"])
            )
            config["selections"] = selections
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))
        torch.hub.set_dir(config["hub_dir"])
        segmenter = (
            DirectPointSegmenter(camera_names) if config["selection"] == "keypoints"
            else SAM3Segmenter(camera_names, device=config["device"], **config["segmenter"])
        )
        tracker = OnlineKeypointTracker(camera_names, device=config["device"], **config["tracker"])
        pipeline = OnlineTrackingPipeline(segmenter, tracker, keep_history=True)
        frames = {camera: [] for camera in camera_names}
        results = {camera: [] for camera in camera_names}
        records, inference_times = [], []
        bundle = first
        with (output_dir / "timings.csv").open("w", newline="") as file:
            ticker = csv.DictWriter(file, fieldnames=["frame_index", "source_index", "inferred_frame_index", "inference_ran", "inference_ms"])
            ticker.writeheader()
            while bundle is not None:
                for camera in camera_names:
                    frames[camera].append(bundle.images[camera].copy())
                if bundle.index == 0:
                    if config["selection"] == "keypoints":
                        pipeline.initialize_points(bundle.images, selections)
                    else:
                        pipeline.initialize(bundle.images, selections)
                        segmenter.visualize(output_dir, show=config["show_initial"])
                else:
                    pipeline.push(bundle.images)
                latest = pipeline.get_latest_keypoints()
                inferred_index = next(iter(latest.values())).frame_index if latest is not None else None
                elapsed = float(sum(tracker.last_inference_ms.values())) if tracker.updated else None
                if tracker.updated and latest is not None:
                    inference_times.append(elapsed)
                    for camera in camera_names:
                        results[camera].append(latest[camera])
                print(f"frame={bundle.index:04d} source={bundle.source_index:04d} inferred_frame={inferred_index} inference_ran={tracker.updated} inference_ms={elapsed}", flush=True)
                ticker.writerow({"frame_index": bundle.index, "source_index": bundle.source_index,
                                 "inferred_frame_index": inferred_index, "inference_ran": tracker.updated,
                                 "inference_ms": elapsed})
                file.flush()
                records.append({"frame_index": bundle.index, "source_index": bundle.source_index})
                bundle = source.next_frame()
        pipeline.finish()
        if pipeline.get_latest_keypoints() is not None:
            pipeline.visualize_current(output_dir)
        videos = _write_videos(output_dir, camera_names, frames, results, config["fps"], config["trail_length"]) if config["save_videos"] else {}
        artifacts = {}
        for camera in camera_names:
            if not results[camera]:
                continue
            npz = output_dir / f"{_safe_name(camera)}_sam3_cotracker.npz"
            np.savez_compressed(npz, tracks=np.stack([item.xy for item in results[camera]]),
                                visibility=np.stack([item.visible for item in results[camera]]),
                                point_ids=results[camera][0].point_ids,
                                object_names=results[camera][0].object_names,
                                frame_indices=np.array([item.frame_index for item in results[camera]]),
                                inference_ms=np.asarray(inference_times))
            artifacts[camera] = {"tracks": str(npz)}
            if camera in videos:
                artifacts[camera]["video"] = videos[camera]
        (output_dir / "frames.json").write_text(json.dumps(records, indent=2))
        (output_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2))
        print(f"Received {len(records)} frames; saved {len(inference_times)} inferred frames to {output_dir}.", flush=True)


if __name__ == "__main__":
    main()
