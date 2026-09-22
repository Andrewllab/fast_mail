"""Run memory-bounded CoTracker3 offline over a recorded RGB-D episode.

Videos can be processed as overlapping windows. Point IDs and object names
never change: each new window is queried with the previous window's predicted
coordinates at the shared boundary. This provides continuous trajectories
without loading the complete RGB or depth recording into memory.
"""

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
from alex_3d.selection import flatten_direct_points, sample_mask_grid
from alex_3d.track_online_pipe import CameraCalibration, lift_keypoints_3d


def offline_sample_indices(
    frame_count: int,
    first_frame: int = 15,
    step: int = 8,
) -> np.ndarray:
    """Return frames persisted from the full offline trajectory."""
    return output_frame_indices(frame_count, first_frame, step)


def _load_predictor(args, device):
    if args.checkpoint is None:
        return torch.hub.load(
            args.cotracker_repo, "cotracker3_offline",
            source=args.hub_source, trust_repo=True,
        ).to(device).eval()
    repository = Path(args.cotracker_repo).resolve()
    if args.hub_source != "local" or not repository.is_dir():
        raise ValueError("--checkpoint requires --hub-source local and a local --cotracker-repo")
    sys.path.insert(0, str(repository))
    from cotracker.predictor import CoTrackerPredictor

    return CoTrackerPredictor(checkpoint=str(args.checkpoint)).to(device).eval()


def _window_settings(frame_count, max_window_frames, overlap_frames):
    if frame_count < 16:
        raise ValueError("offline tracking needs at least 16 frames")
    if max_window_frames is None:
        return frame_count, 0
    window = int(max_window_frames)
    if window < 16:
        raise ValueError("max_window_frames must be at least 16")
    window = min(window, frame_count)
    if window == frame_count:
        return window, 0
    overlap = min(16, window // 2) if overlap_frames is None else int(overlap_frames)
    if overlap < 1 or overlap >= window:
        raise ValueError("window_overlap_frames must satisfy 1 <= overlap < window")
    return window, overlap


def _valid_anchor(tracks, frame_index):
    """Return finite coordinates for every ID, falling back only when needed."""
    anchor = tracks[frame_index].copy()
    for point_index in range(anchor.shape[0]):
        if np.isfinite(anchor[point_index]).all():
            continue
        history = tracks[:frame_index, point_index]
        valid = np.flatnonzero(np.isfinite(history).all(axis=1))
        if not len(valid):
            raise RuntimeError(f"point {point_index} has no finite coordinate for window handoff")
        anchor[point_index] = history[valid[-1]]
    return anchor


def _handoff_queries(tracks, visibility, start, previous_end):
    """Build per-ID ``(local_time,x,y)`` anchors from the shared overlap."""
    fallback = _valid_anchor(tracks, start)
    queries = np.column_stack([
        np.zeros(len(fallback), dtype=np.float32), fallback
    ]).astype(np.float32)
    for point_index in range(len(fallback)):
        candidates = np.flatnonzero(
            visibility[start:previous_end, point_index]
            & np.isfinite(tracks[start:previous_end, point_index]).all(axis=1)
        )
        if not len(candidates):
            continue
        local_time = int(candidates[-1])
        queries[point_index, 0] = local_time
        queries[point_index, 1:] = tracks[start + local_time, point_index]
    return queries


def track_video_in_windows(
    model,
    rgb_frames,
    initial_points: np.ndarray,
    device,
    max_window_frames: int | None = None,
    overlap_frames: int | None = None,
    frame_count: int | None = None,
    progress=None,
):
    """Track an HDF5/NumPy video in bounded overlapping offline calls.

    Returns dense tracks ``[T,N,2]`` and visibility ``[T,N]``. The next
    window uses the previous dense prediction at its start as time-zero query,
    preserving point order and identity without re-detection or re-clicking.
    """
    available_frames = len(rgb_frames)
    frame_count = available_frames if frame_count is None else int(frame_count)
    if frame_count < 1 or frame_count > available_frames:
        raise ValueError("frame_count must lie within the RGB dataset")
    window, overlap = _window_settings(frame_count, max_window_frames, overlap_frames)
    initial_points = np.asarray(initial_points, dtype=np.float32)
    if initial_points.ndim != 2 or initial_points.shape[1] != 2 or not len(initial_points):
        raise ValueError("initial_points must be nonempty [N,2]")
    dense_tracks = np.full((frame_count, len(initial_points), 2), np.nan, np.float32)
    dense_visibility = np.zeros((frame_count, len(initial_points)), bool)
    start = 0
    previous_end = 0
    window_index = 0
    while start < frame_count:
        end = min(start + window, frame_count)
        queries_array = (
            np.column_stack([
                np.zeros(len(initial_points), dtype=np.float32), initial_points
            ]).astype(np.float32)
            if start == 0 else
            _handoff_queries(dense_tracks, dense_visibility, start, previous_end)
        )
        frames = np.asarray(rgb_frames[start:end])
        video = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(device)
        queries = torch.from_numpy(queries_array).unsqueeze(0).to(device)
        with torch.inference_mode():
            tracks, visibility = model(video[None], queries=queries, grid_size=0)
        tracks = tracks[0].detach().cpu().numpy().astype(np.float32, copy=False)
        visibility = visibility[0].detach().cpu().numpy().astype(bool, copy=False)
        expected_tracks = (end - start, len(initial_points), 2)
        if tracks.shape != expected_tracks or visibility.shape != expected_tracks[:2]:
            raise RuntimeError(
                f"CoTracker returned {tracks.shape}/{visibility.shape}, expected "
                f"{expected_tracks}/{expected_tracks[:2]}"
            )
        dense_tracks[start:end] = tracks
        dense_visibility[start:end] = visibility
        window_index += 1
        if progress is not None:
            progress(window_index, start, end, frame_count)
        del video, queries, tracks, visibility, frames
        if end == frame_count:
            break
        previous_end = end
        next_start = end - overlap
        if frame_count - next_start < 16:
            next_start = frame_count - 16
        if next_start <= start:
            raise RuntimeError("offline window configuration did not advance")
        start = next_start
    return dense_tracks, dense_visibility


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--selection", choices=["prompt", "mask_click", "keypoints"], default="keypoints")
    parser.add_argument("--objects", nargs="+", default=["hand"])
    parser.add_argument("--prompt", default="human hand")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grid-spacing", type=int, default=12)
    parser.add_argument("--max-points", type=int, default=32)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-frames", type=int, help="Optional cap on total recording frames")
    parser.add_argument("--max-window-frames", type=int, help="Maximum RGB frames per model call")
    parser.add_argument("--window-overlap-frames", type=int)
    parser.add_argument("--first-output-frame", type=int, default=15)
    parser.add_argument("--output-step", type=int, default=8)
    parser.add_argument("--overwrite-tracking", action="store_true")
    parser.add_argument("--cotracker-repo", default="facebookresearch/co-tracker")
    parser.add_argument("--hub-source", choices=["github", "local"], default="github")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--visualization-fps", type=float)
    args = parser.parse_args(arguments)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    with h5py.File(args.recording, "r") as handle:
        cameras = sorted(handle["obs"].keys())
        recording_fps = recording_frequency_hz(handle)
        counts, first_images, calibrations = {}, {}, {}
        for camera in cameras:
            rgb_dataset = handle[f"obs/{camera}/frames/rgb"]
            count = len(rgb_dataset) if args.max_frames is None else min(
                len(rgb_dataset), args.max_frames
            )
            if count < 16:
                raise ValueError(f"{camera}: offline tracking needs at least 16 frames")
            counts[camera] = count
            first_images[camera] = rgb_dataset[0]
            calibrations[camera] = CameraCalibration(
                handle[f"obs/{camera}/meta/intrinsics"][()],
                handle[f"obs/{camera}/frames/global_extrinsics"][0],
                orthographic=True,
            )
    selections = selections_for_mode(
        args.selection, first_images, args.objects,
        {name: args.prompt for name in args.objects},
    )
    if args.selection == "keypoints":
        selected = flatten_direct_points(first_images, selections, cameras)
    else:
        from alex.track_online_pipe import SAM3Segmenter

        segmenter = SAM3Segmenter(
            cameras, device=str(device), local_files_only=args.local_files_only
        )
        masks = segmenter.segment(first_images, selections)
        segmenter.release_models()
        selected = {
            camera: sample_mask_grid(
                masks[camera], first_images[camera].shape[:2],
                args.grid_spacing, args.max_points,
            )
            for camera in cameras
        }

    model = _load_predictor(args, device)
    for camera in cameras:
        points, names = selected[camera]
        with h5py.File(args.recording, "r") as handle:
            rgb_dataset = handle[f"obs/{camera}/frames/rgb"]
            depth_dataset = handle[f"obs/{camera}/frames/depth"]
            timestamps = handle[f"obs/{camera}/frames/time"][:counts[camera]]
            frame_count = counts[camera]
            tracks, visibility = track_video_in_windows(
                model, rgb_dataset, points, device,
                max_window_frames=args.max_window_frames,
                overlap_frames=args.window_overlap_frames,
                frame_count=frame_count,
                progress=lambda number, start, end, total: print(
                    f"{camera}: offline window {number} frames [{start}, {end})/{total}",
                    flush=True,
                ),
            )
            indices = offline_sample_indices(
                frame_count, args.first_output_frame, args.output_step
            )
            if not len(indices):
                raise ValueError("output sampling selected no frames")
            tracks_output = tracks[indices]
            visibility_output = visibility[indices]
            world, valid = [], []
            for index, xy, visible in zip(indices, tracks_output, visibility_output):
                points_world, depth_valid = lift_keypoints_3d(
                    calibrations[camera], xy, visible, depth_dataset[index]
                )
                world.append(points_world)
                valid.append(depth_valid)
            point_ids = np.arange(len(points))
            playback_fps = selected_visualization_fps(
                timestamps, indices, args.output_step,
                override=args.visualization_fps, recording_fps=recording_fps,
            )
            if args.visualization_dir is not None:
                video_path = write_keypoint_video(
                    visualization_video_path(args.visualization_dir, "offline", camera),
                    rgb_dataset, indices, tracks_output, visibility_output,
                    point_ids, names, playback_fps,
                )
                print(
                    f"{camera}: saved {playback_fps:.3f} FPS video to {video_path}",
                    flush=True,
                )
            if args.visualize:
                play_keypoint_video_frames(
                    f"offline tracking: {camera}", rgb_dataset, indices,
                    tracks_output, visibility_output, point_ids, names, playback_fps,
                )
        write_tracking_group(
            args.recording, "offline", camera,
            tracks_output, np.stack(world), np.stack(valid), indices,
            np.arange(len(points)), names, selections,
            overwrite=args.overwrite_tracking,
        )
        print(
            f"{camera}: stored {len(indices)} offline samples from {frame_count} frames",
            flush=True,
        )
        del tracks, visibility, tracks_output, visibility_output
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
