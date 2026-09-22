"""Run CoTracker3 online over a recorded RGB-D HDF5 episode in-place."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alex_3d.hdf5_io import write_tracking_group
from alex_3d.live_utils import (
    output_frame_indices,
    play_keypoint_video_frames,
    recording_frequency_hz,
    selected_visualization_fps,
    selections_for_mode,
    visualization_video_path,
    write_keypoint_video,
)
from alex_3d.selection import DirectPointSegmenter
from alex_3d.track_online_pipe import CameraCalibration, lift_keypoints_3d


def _calibration(handle, camera):
    intrinsics = handle[f"obs/{camera}/meta/intrinsics"][()]
    extrinsics = handle[f"obs/{camera}/frames/global_extrinsics"][0]
    return CameraCalibration(intrinsics, extrinsics, orthographic=True)


def _tracker_options(args):
    options = {
        "device": args.device,
        "grid_spacing": args.grid_spacing,
        "max_points_per_object": args.max_points,
        "hub_repo": args.cotracker_repo,
        "hub_source": args.hub_source,
    }
    if args.checkpoint is not None:
        repository = Path(args.cotracker_repo).resolve()
        if args.hub_source != "local" or not repository.is_dir():
            raise ValueError("--checkpoint requires --hub-source local and a local --cotracker-repo")
        sys.path.insert(0, str(repository))
        from cotracker.predictor import CoTrackerOnlinePredictor

        options["predictor_factory"] = lambda: CoTrackerOnlinePredictor(
            checkpoint=str(args.checkpoint)
        )
    return options


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--pipeline", choices=["independent", "batch"], default="independent")
    parser.add_argument("--selection", choices=["prompt", "mask_click", "keypoints"], default="keypoints")
    parser.add_argument("--objects", nargs="+", default=["hand"])
    parser.add_argument("--prompt", default="human hand")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grid-spacing", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cotracker-repo", default="facebookresearch/co-tracker")
    parser.add_argument("--hub-source", choices=["github", "local"], default="github")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--overwrite-tracking", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--visualization-fps", type=float)
    parser.add_argument("--first-output-frame", type=int, default=15)
    parser.add_argument("--output-step", type=int, default=8)
    parser.add_argument("--export-3d-dir", type=Path)
    parser.add_argument("--export-3d-fps", type=float, default=2.5)
    args = parser.parse_args(arguments)
    api = __import__(
        "alex_batch_3d.track_online_pipe" if args.pipeline == "batch" else "alex_3d.track_online_pipe",
        fromlist=["SAM3Segmenter", "OnlineKeypointTracker", "OnlineTrackingPipeline"],
    )
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    with h5py.File(args.recording, "r") as handle:
        cameras = sorted(handle["obs"].keys())
        rgbs = {camera: handle[f"obs/{camera}/frames/rgb"][()] for camera in cameras}
        depths = {camera: handle[f"obs/{camera}/frames/depth"][()] for camera in cameras}
        timestamps = {
            camera: handle[f"obs/{camera}/frames/time"][()] for camera in cameras
        }
        recording_fps = recording_frequency_hz(handle)
        calibrations = {camera: _calibration(handle, camera) for camera in cameras}
    lengths = {camera: len(value) for camera, value in rgbs.items()}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values())) < 16:
        raise ValueError(f"all cameras need at least 16 synchronized frames: {lengths}")
    first_images = {camera: value[0] for camera, value in rgbs.items()}
    selections = selections_for_mode(
        args.selection, first_images, args.objects,
        {name: args.prompt for name in args.objects},
    )
    args.device = device
    tracker = api.OnlineKeypointTracker(cameras, **_tracker_options(args))
    segmenter = (
        DirectPointSegmenter(cameras) if args.selection == "keypoints" else
        api.SAM3Segmenter(cameras, device=device, local_files_only=args.local_files_only)
    )
    pipeline = api.OnlineTrackingPipeline(
        segmenter, tracker, calibrations, keep_history=False
    )
    if args.selection == "keypoints":
        pipeline.initialize_points(first_images, selections)
    else:
        pipeline.initialize(first_images, selections)

    for index in range(1, next(iter(lengths.values()))):
        images = {camera: rgbs[camera][index] for camera in cameras}
        depth_images = {camera: depths[camera][index] for camera in cameras}
        pipeline.push(images, depth_images)
        if not tracker.updated:
            continue
    if not tracker.tracks:
        raise RuntimeError("No complete CoTracker window was processed")
    covered_frames = min(len(tracker.tracks[camera]) for camera in cameras)
    frame_indices = output_frame_indices(
        covered_frames, args.first_output_frame, args.output_step
    )
    if not len(frame_indices):
        raise ValueError("output sampling selected no completed CoTracker frames")
    for camera in cameras:
        point_ids, names = tracker.identities[camera]
        keypoints_2d = np.asarray(tracker.tracks[camera])[frame_indices]
        cotracker_visibility = np.asarray(tracker.visibility[camera])[frame_indices]
        keypoints_3d, output_visibility = [], []
        for index, xy, visible in zip(
            frame_indices, keypoints_2d, cotracker_visibility
        ):
            points_world, depth_valid = lift_keypoints_3d(
                calibrations[camera], xy, visible, depths[camera][index]
            )
            keypoints_3d.append(points_world)
            output_visibility.append(depth_valid)
        keypoints_3d = np.stack(keypoints_3d)
        output_visibility = np.stack(output_visibility)
        write_tracking_group(
            args.recording, "online", camera,
            keypoints_2d, keypoints_3d, output_visibility, frame_indices,
            point_ids, names, selections, overwrite=args.overwrite_tracking,
        )
        pipeline.world_tracks[camera] = list(keypoints_3d)
        pipeline.world_visibility[camera] = list(output_visibility)
        pipeline.world_frame_indices[camera] = frame_indices.tolist()
        playback_fps = selected_visualization_fps(
            timestamps[camera], frame_indices, args.output_step,
            override=args.visualization_fps, recording_fps=recording_fps,
        )
        if args.visualization_dir is not None:
            video_path = write_keypoint_video(
                visualization_video_path(args.visualization_dir, "online", camera),
                rgbs[camera], frame_indices, keypoints_2d, cotracker_visibility,
                point_ids, names, playback_fps,
            )
            print(f"{camera}: saved {playback_fps:.3f} FPS video to {video_path}", flush=True)
        if args.visualize:
            play_keypoint_video_frames(
                f"online tracking: {camera}", rgbs[camera], frame_indices,
                keypoints_2d, cotracker_visibility, point_ids, names, playback_fps,
            )
    if args.export_3d_dir is not None:
        exported = pipeline.export_3d(args.export_3d_dir, fps=args.export_3d_fps)
        print(f"Exported 3D trajectory visualization to {args.export_3d_dir}: {exported}")
    print(
        f"Stored {len(frame_indices)} online tracking samples from "
        f"{covered_frames} completed frames in {args.recording}"
    )


if __name__ == "__main__":
    main()
