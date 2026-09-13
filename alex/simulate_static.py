"""Replay synchronized HDF5 camera frames through the online tracking pipeline.

Run with --inspect first. Configuration is DEFAULT_CONFIG plus an optional JSON
file and repeated --set dotted.key=JSON_VALUE overrides; Hydra is not used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import cv2
import h5py
import numpy as np
import torch

if __package__:
    from .track_online_pipe import OnlineKeypointTracker, OnlineTrackingPipeline, SAM3Segmenter, save_rgb, _safe_name
else:
    from track_online_pipe import OnlineKeypointTracker, OnlineTrackingPipeline, SAM3Segmenter, save_rgb, _safe_name


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = {
    "source": {
        "path": "/home/david/projects_wr/2026_05_18-12_49_11.h5",
        "camera_paths": None,
        "start": 0,
        "stop": None,
        "stride": 1,
        "resize_wh": None,
        "color_order": "RGB",
        "float_range": "0_255",
    },
    "device": "auto",
    "cpu_threads": 4,
    "objects": ["object"],
    "selections": None,
    "segmenter": {
        "model_id": "facebook/sam3",
        "cache_dir": str(HERE / ".cache" / "huggingface"),
        "local_files_only": False,
        "threshold": 0.5,
    },
    "tracker": {
        "grid_spacing": 16,
        "max_points_per_object": 128,
        "support_grid": True,
        "hub_repo": "facebookresearch/co-tracker",
        "hub_source": "github",
    },
    "hub_dir": str(HERE / ".cache" / "torch" / "hub"),
    "output_dir": str(HERE / "outputs"),
    "fps": 15.0,
    "save_videos": True,
    "trail_length": 20,
    "show_initial": False,
    "show_live": False,
    "play_final": False,
}


@dataclass
class FrameBundle:
    index: int
    source_index: int
    images: dict[str, np.ndarray]
    original_sizes_wh: dict[str, tuple[int, int]]


class HDF5FrameReader:
    """Read RGB views at the current row, then advance the pointer.

    No prefetch, timestamps, actions, depth or playback pacing are used.
    next_frame() returns None at EOF. Images are contiguous RGB uint8 HWC.
    """

    def __init__(
        self, path: str, camera_paths: Mapping[str, str] | None = None,
        start: int = 0, stop: int | None = None, stride: int = 1,
        resize_wh: list[int] | None = None,
        color_order: str = "RGB", float_range: str = "0_255",
    ):
        if start < 0 or stride < 1:
            raise ValueError("start must be nonnegative and stride must be positive")
        if resize_wh is not None and (len(resize_wh) != 2 or min(resize_wh) < 2):
            raise ValueError("resize_wh must be [width,height] with both >= 2")
        if color_order not in ("RGB", "BGR") or float_range not in ("0_255", "0_1"):
            raise ValueError("Unsupported color_order or float_range")
        self.file = h5py.File(path, "r")
        try:
            self.camera_paths = dict(camera_paths) if camera_paths is not None else self.discover_cameras(self.file)
            if not self.camera_paths:
                raise ValueError("No image streams found; configure source.camera_paths explicitly")
            self.datasets = {camera: self.file[dataset] for camera, dataset in self.camera_paths.items()}
            lengths = {camera: len(dataset) for camera, dataset in self.datasets.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"Camera lengths differ; align recordings first: {lengths}")
            self.total_frames = next(iter(lengths.values()))
            stop = self.total_frames if stop is None else min(stop, self.total_frames)
            self.indices = range(start, stop, stride)
            if not self.indices:
                raise ValueError("Selected source interval is empty")
            self.resize_wh = tuple(resize_wh) if resize_wh is not None else None
            self.color_order = color_order
            self.float_range = float_range
            self.pointer = 0
            self.closed = False
        except Exception:
            self.file.close()
            raise

    @staticmethod
    def discover_cameras(handle: h5py.File):
        paths = {}
        if "obs" not in handle:
            return paths
        for camera, group in handle["obs"].items():
            if not isinstance(group, h5py.Group) or "frames" not in group:
                continue
            for stream in ("rgb", "left"):
                if stream in group["frames"]:
                    dataset = group["frames"][stream]
                    if dataset.ndim in (3, 4):
                        paths[camera] = dataset.name.lstrip("/")
                        break
        return paths

    @staticmethod
    def preprocess_image(image, resize_wh=None, color_order="RGB", float_range="0_255"):
        if color_order not in ("RGB", "BGR") or float_range not in ("0_255", "0_1"):
            raise ValueError("Unsupported color_order or float_range")
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(f"Expected HWC RGB/RGBA or HW grayscale, got {image.shape}")
        image = image[..., :3]
        if np.issubdtype(image.dtype, np.floating):
            maximum = 1.0 if float_range == "0_1" else 255.0
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > maximum:
                raise ValueError(f"Image values must lie in [0,{maximum}]")
            image = np.round(image * (255.0 if float_range == "0_1" else 1.0)).astype(np.uint8)
        elif image.dtype != np.uint8:
            raise ValueError(f"Unsupported image dtype: {image.dtype}; convert explicitly to uint8")
        if color_order == "BGR":
            image = image[..., ::-1]
        if resize_wh is not None:
            image = cv2.resize(image, tuple(resize_wh), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(image).copy()

    def next_frame(self) -> FrameBundle | None:
        if self.closed:
            raise RuntimeError("Frame reader is closed")
        if self.pointer >= len(self.indices):
            return None
        source_index = self.indices[self.pointer]
        images, original_sizes = {}, {}
        for camera, dataset in self.datasets.items():
            raw = dataset[source_index]
            original_sizes[camera] = (raw.shape[1], raw.shape[0])
            images[camera] = self.preprocess_image(raw, self.resize_wh, self.color_order, self.float_range)
        bundle = FrameBundle(self.pointer, source_index, images, original_sizes)
        self.pointer += 1
        return bundle

    def __len__(self):
        return len(self.indices) - self.pointer

    def close(self):
        self.file.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


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
    parser.add_argument("--config", type=Path, help="JSON overrides for DEFAULT_CONFIG")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--inspect", action="store_true", help="Inspect HDF5 and save first-frame previews without loading models")
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
    return config, args.inspect


def _write_sparse_videos(output_dir, camera_names, frames, results, fps, trail_length):
    """Write one video frame per completed CoTracker window."""
    paths = {}
    for camera in camera_names:
        camera_results = results[camera]
        if not camera_results:
            continue
        video_path = Path(output_dir) / f"{_safe_name(camera)}_sam3_cotracker.mp4"
        height, width = frames[camera][0].shape[:2]
        encoded_size = (width + width % 2, height + height % 2)
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, encoded_size)
        if not writer.isOpened():
            raise OSError(f"Video encoder could not open {video_path}")
        try:
            previous_results = []
            for result in camera_results:
                source_frame = frames[camera][result.frame_index].copy()
                recent = previous_results[-trail_length:] if trail_length > 0 else []
                for previous in recent:
                    for point_index, (point, visible) in enumerate(zip(result.xy, result.visible)):
                        if visible and previous.visible[point_index]:
                            start = tuple(np.round(previous.xy[point_index]).astype(int))
                            end = tuple(np.round(point).astype(int))
                            cv2.line(source_frame, start, end, (255, 180, 0), 1, cv2.LINE_AA)
                overlay = OnlineKeypointTracker.visualize({camera: source_frame}, {camera: result})[camera]
                overlay = cv2.copyMakeBorder(overlay, 0, height % 2, 0, width % 2,
                                             cv2.BORDER_CONSTANT)
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                previous_results.append(result)
        finally:
            writer.release()
        paths[camera] = str(video_path)
    return paths


def main(arguments=None):
    config, inspect_only = read_config(arguments)
    if config["fps"] <= 0 or config["cpu_threads"] < 1:
        raise ValueError("fps and cpu_threads must be positive")
    torch.set_num_threads(config["cpu_threads"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with HDF5FrameReader(**config["source"]) as source:
        first = source.next_frame()
        if first is None:
            raise RuntimeError("The selected source interval contains no frames")
        camera_names = list(first.images)
        print(f"Source: {config['source']['path']} | {source.total_frames} frames | cameras: {camera_names}", flush=True)
        for camera, frame in first.images.items():
            print(f"  {camera}: {source.camera_paths[camera]} -> {frame.shape}, {frame.dtype}", flush=True)
        for camera, frame in first.images.items():
            save_rgb(output_dir / f"{_safe_name(camera)}_first.png", frame)
        if inspect_only:
            print(f"First-frame previews saved in {output_dir}", flush=True)
            return
        torch.hub.set_dir(config["hub_dir"])
        selections = config["selections"]
        if selections is None:
            selections = SAM3Segmenter.collect_clicks(first.images, config["objects"])
            config["selections"] = selections
        (output_dir / "config.json").write_text(json.dumps(config, indent=2))
        segmenter = SAM3Segmenter(camera_names, device=config["device"], **config["segmenter"])
        tracker = OnlineKeypointTracker(camera_names, device=config["device"], **config["tracker"])
        pipeline = OnlineTrackingPipeline(segmenter, tracker, keep_history=True)
        records = []
        all_frames = {camera: [] for camera in camera_names}
        inferred_results = {camera: [] for camera in camera_names}
        inference_times = []
        bundle = first
        with (output_dir / "timings.csv").open("w", newline="") as timing_file:
            fields = ["frame_index", "source_index", "inferred_frame_index", "inference_ran", "inference_ms"]
            ticker = csv.DictWriter(timing_file, fieldnames=fields)
            ticker.writeheader()
            while bundle is not None:
                for camera in camera_names:
                    all_frames[camera].append(bundle.images[camera].copy())
                if bundle.index == 0:
                    pipeline.initialize(bundle.images, selections)
                    segmenter.visualize(output_dir, show=config["show_initial"])
                else:
                    pipeline.push(bundle.images)
                result = pipeline.get_latest_keypoints()
                inferred_index = next(iter(result.values())).frame_index if result is not None else None
                counts = {camera: f"{int(value.visible.sum())}/{len(value.xy)}" for camera, value in result.items()} if result is not None else {}
                inference_ms = sum(tracker.last_inference_ms.values()) if tracker.updated else None
                if tracker.updated and result is not None:
                    inference_times.append(float(inference_ms))
                    for camera in camera_names:
                        inferred_results[camera].append(result[camera])
                print(f"frame={bundle.index:04d} source={bundle.source_index:04d} inferred_frame={inferred_index} inference_ran={tracker.updated} inference_ms={inference_ms} visible={counts}", flush=True)
                stop_requested = False
                if config["show_live"]:
                    for camera, overlay in pipeline.visualize_current().items():
                        cv2.imshow(camera, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                    stop_requested = cv2.waitKey(1) & 0xFF in (27, ord("q"))
                ticker.writerow(dict(frame_index=bundle.index, source_index=bundle.source_index,
                                     inferred_frame_index=inferred_index, inference_ran=tracker.updated,
                                     inference_ms=inference_ms))
                timing_file.flush()
                records.append({"frame_index": bundle.index, "source_index": bundle.source_index,
                                "original_sizes_wh": bundle.original_sizes_wh})
                if stop_requested:
                    break
                bundle = source.next_frame()
        pipeline.finish()
        if pipeline.get_latest_keypoints() is not None:
            pipeline.visualize_current(output_dir)
        paths = {}
        if config["save_videos"]:
            paths = _write_sparse_videos(output_dir, camera_names, all_frames, inferred_results,
                                         config["fps"], config["trail_length"])
        sparse_tracks = {}
        for camera in camera_names:
            camera_results = inferred_results[camera]
            if not camera_results:
                continue
            npz_path = output_dir / f"{_safe_name(camera)}_sam3_cotracker.npz"
            np.savez_compressed(
                npz_path,
                tracks=np.stack([item.xy for item in camera_results]),
                visibility=np.stack([item.visible for item in camera_results]),
                point_ids=camera_results[0].point_ids,
                object_names=camera_results[0].object_names,
                frame_indices=np.array([item.frame_index for item in camera_results]),
                inference_ms=np.asarray(inference_times, dtype=np.float64),
            )
            sparse_tracks[camera] = str(npz_path)
        paths = {
            camera: {
                **({"video": paths[camera]} if camera in paths else {}),
                "tracks": sparse_tracks[camera],
            }
            for camera in camera_names if camera in sparse_tracks
        }
        (output_dir / "frames.json").write_text(json.dumps(records, indent=2))
        (output_dir / "artifacts.json").write_text(json.dumps(paths, indent=2))
        cv2.destroyAllWindows()
        tracked_count = len(inferred_results[camera_names[0]])
        print(f"Received {len(records)} frames; saved {tracked_count} inferred frames to {output_dir}. Unprocessed tail: {len(records) - tracked_count}.", flush=True)
        if config["play_final"]:
            for camera_paths in paths.values():
                if "video" in camera_paths:
                    pipeline.play_video(camera_paths["video"])


if __name__ == "__main__":
    main()
