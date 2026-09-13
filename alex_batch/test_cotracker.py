"""Run batched CoTracker3 on the example episode without requiring SAM3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from alex.simulate_static import HDF5FrameReader
from alex.track_online_pipe import _safe_name
if __package__:
    from .track_online_pipe import OnlineKeypointTracker, OnlineTrackingPipeline, SAM3Segmenter
else:
    from track_online_pipe import OnlineKeypointTracker, OnlineTrackingPipeline, SAM3Segmenter


class DiagnosticMaskSegmenter(SAM3Segmenter):
    BOXES = {"left_cam": (151, 65, 184, 105), "right_cam": (77, 92, 110, 125)}

    def segment(self, images, selections):
        self.initial_images = {camera: frame.copy() for camera, frame in images.items()}
        self.masks = {}
        for camera, frame in images.items():
            left, top, right, bottom = self.BOXES[camera]
            mask = np.zeros(frame.shape[:2], dtype=bool)
            mask[top:bottom, left:right] = True
            self.masks[camera] = {"manual_roi": mask}
        return self.masks


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default="/home/david/projects_wr/2026_05_18-12_49_11.h5")
    parser.add_argument("--output", default="alex_batch/outputs/cotracker_episode")
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args(arguments)
    cameras = ("left_cam", "right_cam")
    torch.hub.set_dir(str(Path(__file__).resolve().parent / ".cache/torch/hub"))
    camera_paths = {camera: f"obs/{camera}/frames/left" for camera in cameras}
    tracker = OnlineKeypointTracker(cameras, device="auto", grid_spacing=12,
                                    max_points_per_object=12)
    pipeline = OnlineTrackingPipeline(DiagnosticMaskSegmenter(cameras, device=tracker.device), tracker)
    frames, inferred, timings = {camera: [] for camera in cameras}, {camera: [] for camera in cameras}, []
    with HDF5FrameReader(args.path, camera_paths=camera_paths) as source:
        first = source.next_frame()
        pipeline.initialize(first.images, {})
        bundle = first
        while bundle is not None:
            for camera in cameras:
                frames[camera].append(bundle.images[camera].copy())
            if bundle.index:
                pipeline.push(bundle.images)
            if tracker.updated:
                result = pipeline.get_latest_keypoints()
                for camera in cameras:
                    inferred[camera].append(result[camera])
                timings.append(float(sum(tracker.last_inference_ms.values())))
            bundle = source.next_frame()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    videos = {}
    for camera in cameras:
        if not inferred[camera]:
            continue
        height, width = frames[camera][0].shape[:2]
        path = output / f"{_safe_name(camera)}_batched_cotracker.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                 (width + width % 2, height + height % 2))
        for result in inferred[camera]:
            image = frames[camera][result.frame_index]
            overlay = OnlineKeypointTracker.visualize({camera: image}, {camera: result}, show_ids=True)[camera]
            writer.write(cv2.cvtColor(cv2.copyMakeBorder(overlay, 0, height % 2, 0, width % 2,
                                                         cv2.BORDER_CONSTANT), cv2.COLOR_RGB2BGR))
        writer.release()
        videos[camera] = str(path)
    (output / "report.json").write_text(json.dumps({
        "source": args.path, "total_received_frames": len(frames[cameras[0]]),
        "inferred_frame_indices": [item.frame_index for item in inferred[cameras[0]]],
        "batched_inference_calls": tracker.inference_count, "inference_ms": timings, "videos": videos,
    }, indent=2))
    print(f"Received {len(frames[cameras[0]])} frames; wrote {len(timings)} inference frames to {output}")


if __name__ == "__main__":
    main()
