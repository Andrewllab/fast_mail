"""Run CoTracker3 online over a complete episode and save sparse outputs.

This diagnostic deliberately uses manual first-frame rectangles instead of SAM3.
The output video contains only frames on which CoTracker inferred a result:
15, 23, 31, ... for a 16-frame window and eight-frame stride. Buffered frames
are consumed but never duplicated into the output video.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__:
    from .simulate_static import HDF5FrameReader
    from .track_online_pipe import OnlineKeypointTracker, OnlineTrackingPipeline, SAM3Segmenter, _safe_name
else:
    from simulate_static import HDF5FrameReader
    from track_online_pipe import OnlineKeypointTracker, OnlineTrackingPipeline, SAM3Segmenter, _safe_name


class DiagnosticMaskSegmenter(SAM3Segmenter):
    """Use fixed rectangles to test CoTracker without requiring SAM3 weights."""

    BOXES = {"left_cam": (151, 65, 184, 105), "right_cam": (77, 92, 110, 125)}

    def segment(self, images, selections):
        self.initial_images = images
        self.masks = {}
        for camera, frame in images.items():
            if camera not in self.BOXES:
                raise ValueError(f"No diagnostic rectangle configured for {camera}")
            left, top, right, bottom = self.BOXES[camera]
            mask = np.zeros(frame.shape[:2], dtype=bool)
            mask[top:bottom, left:right] = True
            self.masks[camera] = {"manual_red_cup_roi": mask}
        return self.masks


def _write_sparse_videos(output, camera_names, frames, results, fps):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for camera in camera_names:
        if not results[camera]:
            continue
        first = frames[camera][0]
        height, width = first.shape[:2]
        path = output / f"{_safe_name(camera)}_inference_frames.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise OSError(f"Could not open video writer: {path}")
        try:
            for image, result in zip(frames[camera], results[camera]):
                overlay = OnlineKeypointTracker.visualize(
                    {camera: image}, {camera: result}, show_ids=True
                )[camera]
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        artifacts[camera] = str(path)
    return artifacts


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="/home/david/projects_wr/2026_05_18-12_49_11.h5")
    parser.add_argument("--repo", default=".cache/torch/hub/facebookresearch_co-tracker_main")
    parser.add_argument("--output", default="alex/outputs/cotracker_episode")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cameras", nargs="+", default=["left_cam", "right_cam"], choices=["left_cam", "right_cam"])
    args = parser.parse_args(arguments)
    if args.fps <= 0 or args.threads < 1:
        raise ValueError("fps and threads must be positive")
    torch.set_num_threads(args.threads)
    torch.hub.set_dir(str(Path(__file__).resolve().parent / ".cache/torch/hub"))
    camera_paths = {camera: f"obs/{camera}/frames/left" for camera in args.cameras}
    tracker = OnlineKeypointTracker(
        args.cameras, device="cuda" if torch.cuda.is_available() else "cpu",
        grid_spacing=12, max_points_per_object=12,
        hub_repo=args.repo, hub_source="local",
    )
    pipeline = OnlineTrackingPipeline(
        DiagnosticMaskSegmenter(args.cameras, device=tracker.device), tracker
    )
    frames = {camera: [] for camera in args.cameras}
    inferred = {camera: [] for camera in args.cameras}
    inference_times = []
    with HDF5FrameReader(args.path, camera_paths=camera_paths) as source:
        first = source.next_frame()
        if first is None:
            raise RuntimeError("Episode contains no frames")
        pipeline.initialize(first.images, {})
        while True:
            for camera in args.cameras:
                frames[camera].append(first.images[camera])
            bundle = source.next_frame()
            if bundle is None:
                break
            pipeline.push(bundle.images)
            if tracker.updated:
                result = pipeline.get_latest_keypoints()
                for camera in args.cameras:
                    inferred[camera].append(result[camera])
                inference_times.append(sum(tracker.last_inference_ms.values()))
                print(
                    f"inferred frame={next(iter(result.values())).frame_index} "
                    f"inference_ms={inference_times[-1]:.2f}", flush=True
                )
            first = bundle
    pipeline.finish()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # Keep exactly the source image corresponding to each inferred result.
    sparse_frames = {
        camera: [frames[camera][result.frame_index] for result in inferred[camera]]
        for camera in args.cameras
    }
    videos = _write_sparse_videos(output, args.cameras, sparse_frames, inferred, args.fps)
    report = {
        "source": args.path,
        "total_received_frames": len(frames[args.cameras[0]]),
        "inferred_frame_indices": [result.frame_index for result in inferred[args.cameras[0]]],
        "inference_calls_per_camera": tracker.inference_count,
        "inference_ms": inference_times,
        "videos": videos,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(
        f"Received {report['total_received_frames']} frames; wrote "
        f"{len(report['inferred_frame_indices'])} inference frames to {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
