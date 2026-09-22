"""Online 2D CoTracker tracking with RGB-D world-coordinate lifting.

This module keeps the independent CoTracker predictor per camera used by
``alex``.  3D points are produced only when an online window completes: the
last 2D prediction is sampled in the synchronized depth image and unprojected
with that camera's intrinsics and camera-to-world extrinsics.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import cv2
import numpy as np

from alex.track_online_pipe import (
    OnlineKeypointTracker as _OnlineKeypointTracker,
    OnlineTrackingPipeline as _OnlineTrackingPipeline,
    SAM3Segmenter,
    TrackFrame,
    _color,
    _safe_name,
    validate_images,
)


@dataclass(frozen=True)
class CameraCalibration:
    """Calibration for one RGB-D camera.

    ``extrinsics`` maps camera coordinates to world coordinates.  ``depth``
    values are multiplied by ``depth_scale`` before unprojection.
    """

    intrinsics: np.ndarray
    extrinsics: np.ndarray
    depth_scale: float = 1.0
    orthographic: bool = False

    def __post_init__(self):
        intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        extrinsics = np.asarray(self.extrinsics, dtype=np.float64)
        if intrinsics.shape != (3, 3):
            raise ValueError(f"intrinsics must be [3,3], got {intrinsics.shape}")
        if extrinsics.shape != (4, 4):
            raise ValueError(f"extrinsics must be [4,4], got {extrinsics.shape}")
        if not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
            raise ValueError("calibration matrices must be finite")
        if self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")
        object.__setattr__(self, "intrinsics", intrinsics)
        object.__setattr__(self, "extrinsics", extrinsics)

    @classmethod
    def from_mapping(cls, value: "CameraCalibration | Mapping") -> "CameraCalibration":
        if isinstance(value, cls):
            return value
        return cls(
            intrinsics=value["intrinsics"],
            extrinsics=value.get("extrinsics", np.eye(4)),
            depth_scale=float(value.get("depth_scale", 1.0)),
            orthographic=bool(value.get("orthographic", False)),
        )


def lift_keypoints_3d(
    calibration: CameraCalibration,
    xy: np.ndarray,
    visible: np.ndarray,
    depth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift one set of 2D pixels with aligned depth into world coordinates."""
    xy = np.asarray(xy, dtype=np.float64)
    visible = np.asarray(visible, dtype=bool)
    depth = np.asarray(depth)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must be [N,2], got {xy.shape}")
    if visible.shape != (len(xy),):
        raise ValueError("visible must have one value per keypoint")
    if depth.ndim != 2:
        raise ValueError(f"depth must be [H,W], got {depth.shape}")

    height, width = depth.shape
    pixels = np.rint(xy).astype(np.int64)
    valid = visible & np.isfinite(xy).all(axis=1)
    valid &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
    valid &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    clipped_x = np.clip(pixels[:, 0], 0, max(width - 1, 0))
    clipped_y = np.clip(pixels[:, 1], 0, max(height - 1, 0))
    metric_depth = depth[clipped_y, clipped_x].astype(np.float64) * calibration.depth_scale
    valid &= np.isfinite(metric_depth) & (metric_depth > 0)

    points_camera = np.full((len(xy), 3), np.nan, dtype=np.float64)
    if valid.any():
        fx, fy = calibration.intrinsics[0, 0], calibration.intrinsics[1, 1]
        cx, cy = calibration.intrinsics[0, 2], calibration.intrinsics[1, 2]
        u = pixels[valid, 0].astype(np.float64)
        v = pixels[valid, 1].astype(np.float64)
        z = metric_depth[valid]
        x_over_z = (u - cx) / fx
        y_over_z = (v - cy) / fy
        if not calibration.orthographic:
            z = z / np.sqrt(1.0 + x_over_z**2 + y_over_z**2)
        points_camera[valid] = np.stack([x_over_z * z, y_over_z * z, z], axis=-1)

    rotation = calibration.extrinsics[:3, :3]
    translation = calibration.extrinsics[:3, 3]
    finite = np.isfinite(points_camera).all(axis=1)
    points_world = points_camera.copy()
    if finite.any():
        points_world[finite] = points_camera[finite] @ rotation.T + translation
    return points_world.astype(np.float32), valid


class OnlineKeypointTracker(_OnlineKeypointTracker):
    """The original independent-per-camera tracker, re-exported for symmetry."""


class OnlineTrackingPipeline(_OnlineTrackingPipeline):
    """Independent-camera online tracking plus RGB-D 3D trajectory export."""

    def __init__(
        self,
        segmenter: SAM3Segmenter,
        tracker: OnlineKeypointTracker,
        calibrations: Mapping[str, CameraCalibration | Mapping],
        keep_history: bool = True,
    ):
        super().__init__(segmenter, tracker, keep_history=keep_history)
        if set(calibrations) != set(tracker.camera_names):
            raise ValueError("calibrations must contain exactly the tracker cameras")
        self.calibrations = {
            camera: CameraCalibration.from_mapping(calibrations[camera])
            for camera in tracker.camera_names
        }
        self.world_tracks: dict[str, list[np.ndarray]] = {
            camera: [] for camera in tracker.camera_names
        }
        self.world_visibility: dict[str, list[np.ndarray]] = {
            camera: [] for camera in tracker.camera_names
        }
        self.world_frame_indices: dict[str, list[int]] = {
            camera: [] for camera in tracker.camera_names
        }

    def reset_3d(self):
        """Clear accumulated world-frame predictions without resetting CoTracker."""
        for camera in self.tracker.camera_names:
            self.world_tracks[camera].clear()
            self.world_visibility[camera].clear()
            self.world_frame_indices[camera].clear()

    def lift_keypoints_3d(
        self,
        camera: str,
        xy: np.ndarray,
        visible: np.ndarray,
        depth: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Lift one camera's final 2D keypoints into world coordinates.

        Returns ``(points_world [N,3], valid [N])``.  A point is valid only if
        CoTracker marked it visible, its pixel is in bounds, and its depth is
        finite and positive.  This method does not triangulate cameras.
        """
        if camera not in self.calibrations:
            raise KeyError(camera)
        return lift_keypoints_3d(self.calibrations[camera], xy, visible, depth)

    def _record_latest_3d(self, results: Mapping[str, TrackFrame], depths):
        if depths is None:
            raise ValueError("depth_images are required when a CoTracker window completes")
        if set(depths) != set(self.tracker.camera_names):
            raise ValueError("depth_images must contain exactly the tracker cameras")
        for camera, result in results.items():
            points, valid = self.lift_keypoints_3d(
                camera, result.xy, result.visible, depths[camera]
            )
            self.world_tracks[camera].append(points)
            self.world_visibility[camera].append(valid)
            self.world_frame_indices[camera].append(int(result.frame_index))

    def initialize(self, images, selections, depth_images=None, release_segmenter: bool = True):
        """Initialize segmentation/tracking; depth is first used at frame 15."""
        return super().initialize(images, selections, release_segmenter=release_segmenter)

    def initialize_points(self, images, selections):
        """Initialize CoTracker with exact user-selected pixels, skipping SAM3."""
        from alex_3d.selection import initialize_tracker_from_points

        if self.tracker.frame_index >= 0:
            raise RuntimeError("Create a new pipeline for a new episode")
        initialize_tracker_from_points(self.tracker, images, selections)
        self.direct_point_selections = copy.deepcopy(selections)
        return self._record(images, None)

    def push(self, images, depth_images=None):
        """Push synchronized RGB-D frames and lift only completed-window outputs."""
        result = super().push(images)
        if self.tracker.updated:
            self._record_latest_3d(result, depth_images)
        return result

    def get_latest_3d_keypoints(self, camera: str | None = None):
        """Return the latest lifted point set, or all camera sets."""
        if camera is not None:
            if camera not in self.world_tracks:
                raise KeyError(camera)
            if not self.world_tracks[camera]:
                return None
            return {
                "frame_index": self.world_frame_indices[camera][-1],
                "points": self.world_tracks[camera][-1].copy(),
                "visible": self.world_visibility[camera][-1].copy(),
            }
        if not any(self.world_tracks.values()):
            return None
        return {
            name: self.get_latest_3d_keypoints(name)
            for name in self.tracker.camera_names
        }

    def _plot_3d_snapshot(self, camera: str, index: int, output_path: Path | None = None):
        tracks = np.asarray(self.world_tracks[camera])
        visible = np.asarray(self.world_visibility[camera])
        points = tracks[index]
        fig = plt.figure(figsize=(7, 6))
        axis = fig.add_subplot(111, projection="3d")
        names = self.tracker.identities[camera][1]
        for point_index, name in enumerate(names):
            valid = visible[: index + 1, point_index]
            if valid.any():
                path = tracks[: index + 1, point_index][valid]
                color = np.asarray(_color(f"{name}/{point_index}")) / 255.0
                axis.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=1)
                if visible[index, point_index]:
                    axis.scatter(*points[point_index], color=color, s=18)
        axis.set_title(f"{camera} world keypoints, frame {self.world_frame_indices[camera][index]}")
        axis.set_xlabel("world X")
        axis.set_ylabel("world Y")
        axis.set_zlabel("world Z")
        fig.tight_layout()
        if output_path is not None:
            fig.savefig(output_path, dpi=140)
            plt.close(fig)
        return fig

    def export_3d(self, output_dir: str | Path, fps: float = 5.0, trail_length: int | None = None):
        """Export per-camera world tracks, snapshots, and keypoint-only MP4s.

        One video frame corresponds to one completed CoTracker window, normally
        input frames 15, 23, 31, ...; no RGB images are included.
        """
        if fps <= 0:
            raise ValueError("fps must be positive")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        exported = {}
        for camera in self.tracker.camera_names:
            if not self.world_tracks[camera]:
                continue
            tracks = np.asarray(self.world_tracks[camera], dtype=np.float32)
            visible = np.asarray(self.world_visibility[camera], dtype=bool)
            frame_indices = np.asarray(self.world_frame_indices[camera], dtype=np.int64)
            stem = _safe_name(camera)
            npz_path = output_dir / f"{stem}_world_tracks.npz"
            np.savez_compressed(
                npz_path, tracks=tracks, visibility=visible,
                frame_indices=frame_indices,
                point_ids=self.tracker.identities[camera][0],
                object_names=self.tracker.identities[camera][1],
            )
            snapshot = output_dir / f"{stem}_world_final.png"
            self._plot_3d_snapshot(camera, len(tracks) - 1, snapshot)
            video_path = output_dir / f"{stem}_world_keypoints.mp4"
            valid_points = tracks[np.isfinite(tracks).all(axis=-1)]
            if len(valid_points) == 0:
                continue
            low = valid_points.min(axis=0)
            high = valid_points.max(axis=0)
            span = np.maximum(high - low, 1e-3)
            margin = 0.1 * span
            fig = plt.figure(figsize=(7, 6), dpi=120)
            axis = fig.add_subplot(111, projection="3d")
            canvas_width, canvas_height = fig.canvas.get_width_height()
            writer = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                (canvas_width, canvas_height),
            )
            if not writer.isOpened():
                plt.close(fig)
                raise OSError(f"Could not open 3D video writer: {video_path}")
            try:
                for index in range(len(tracks)):
                    axis.clear()
                    names = self.tracker.identities[camera][1]
                    for point_index, name in enumerate(names):
                        start = max(0, index - trail_length + 1) if trail_length else 0
                        indices = np.arange(start, index + 1)
                        indices = indices[visible[indices, point_index]]
                        if len(indices):
                            path = tracks[indices, point_index]
                            color = np.asarray(_color(f"{name}/{point_index}")) / 255.0
                            axis.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=1)
                            axis.scatter(*tracks[index, point_index], color=color, s=18)
                    axis.set_xlim(low[0] - margin[0], high[0] + margin[0])
                    axis.set_ylim(low[1] - margin[1], high[1] + margin[1])
                    axis.set_zlim(low[2] - margin[2], high[2] + margin[2])
                    axis.set_title(f"{camera}: input frame {frame_indices[index]}")
                    axis.set_xlabel("world X")
                    axis.set_ylabel("world Y")
                    axis.set_zlabel("world Z")
                    fig.tight_layout()
                    fig.canvas.draw()
                    rgba = np.asarray(fig.canvas.buffer_rgba())
                    writer.write(cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            plt.close(fig)
            exported[camera] = {"tracks": str(npz_path), "snapshot": str(snapshot), "video": str(video_path)}
        return exported

    def export(self, output_dir: str | Path, *args, export_3d: bool = True, **kwargs):
        """Keep the original 2D export and optionally add world-frame outputs."""
        paths = super().export(output_dir, *args, **kwargs)
        if export_3d:
            paths["3d"] = self.export_3d(output_dir)
        return paths


__all__ = ["CameraCalibration", "lift_keypoints_3d", "SAM3Segmenter", "OnlineKeypointTracker", "OnlineTrackingPipeline", "TrackFrame"]
