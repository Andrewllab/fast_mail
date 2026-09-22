"""HDF5 storage shared by live capture and post-processing tools."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import time
from typing import Mapping

import h5py
import numpy as np

from alex_3d.cameras import RGBDFrame


def _stack(items, name):
    if not items:
        raise ValueError(f"cannot save empty {name}")
    return np.stack(items)


def _string_array(values):
    return np.asarray(values, dtype=h5py.string_dtype("utf-8"))


@dataclass
class RGBDRecording:
    camera_name: str
    calibration: object
    rgb: list[np.ndarray] = field(default_factory=list)
    depth: list[np.ndarray] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    hardware_timestamps_ms: list[float] = field(default_factory=list)
    source_indices: list[int] = field(default_factory=list)
    camera_to_world: list[np.ndarray] = field(default_factory=list)

    def append(self, frame: RGBDFrame):
        self.rgb.append(frame.rgb.copy())
        self.depth.append(frame.depth.copy())
        self.timestamps.append(float(frame.timestamp))
        self.hardware_timestamps_ms.append(
            np.nan if frame.hardware_timestamp_ms is None else float(frame.hardware_timestamp_ms)
        )
        self.source_indices.append(int(frame.index))
        self.camera_to_world.append(frame.camera_to_world.copy())

    def clear(self):
        self.rgb.clear()
        self.depth.clear()
        self.timestamps.clear()
        self.hardware_timestamps_ms.clear()
        self.source_indices.clear()
        self.camera_to_world.clear()

    def __len__(self):
        return len(self.rgb)


@dataclass
class TrackedRGBDRecording(RGBDRecording):
    keypoints_2d: list[np.ndarray] = field(default_factory=list)
    keypoints_3d: list[np.ndarray] = field(default_factory=list)
    visibility: list[np.ndarray] = field(default_factory=list)
    tracker_frame_indices: list[int] = field(default_factory=list)
    point_ids: np.ndarray | None = None
    object_names: np.ndarray | None = None
    selection: dict | None = None

    def append_tracking(self, frame, result_2d, result_3d):
        super().append(frame)
        self.keypoints_2d.append(np.asarray(result_2d.xy, dtype=np.float32).copy())
        self.keypoints_3d.append(np.asarray(result_3d["points"], dtype=np.float32).copy())
        self.visibility.append(
            np.asarray(result_2d.visible, dtype=bool)
            & np.asarray(result_3d["visible"], dtype=bool)
        )
        self.tracker_frame_indices.append(int(result_2d.frame_index))
        if self.point_ids is None:
            self.point_ids = np.asarray(result_2d.point_ids).copy()
            self.object_names = np.asarray(result_2d.object_names).astype(str)

    def clear(self):
        super().clear()
        self.keypoints_2d.clear()
        self.keypoints_3d.clear()
        self.visibility.clear()
        self.tracker_frame_indices.clear()
        self.point_ids = None
        self.object_names = None
        self.selection = None


def _write_rgbd(group: h5py.Group, recording: RGBDRecording):
    frames = group.require_group("frames")
    frames.create_dataset("rgb", data=_stack(recording.rgb, "rgb"), compression="gzip")
    frames.create_dataset("depth", data=_stack(recording.depth, "depth"), compression="gzip")
    frames.create_dataset("time", data=np.asarray(recording.timestamps, dtype=np.float64))
    frames.create_dataset(
        "hardware_time_ms", data=np.asarray(recording.hardware_timestamps_ms, dtype=np.float64)
    )
    frames.create_dataset("source_index", data=np.asarray(recording.source_indices, dtype=np.int64))
    frames.create_dataset(
        "global_extrinsics",
        data=_stack(recording.camera_to_world, "camera_to_world").astype(np.float32),
    )
    meta = group.require_group("meta")
    meta.create_dataset(
        "intrinsics", data=np.asarray(recording.calibration.intrinsics, dtype=np.float32)
    )
    meta.create_dataset(
        "intrinsics_heights_width",
        data=np.asarray(recording.rgb[0].shape[:2], dtype=np.int32),
    )
    meta.attrs["depth_units"] = "metres_optical_axis_z"
    meta.attrs["coordinate_convention"] = "opencv_camera_right_down_forward"


def save_recording(
    path: str | Path,
    recordings: Mapping[str, RGBDRecording],
    metadata: Mapping | None = None,
) -> Path:
    """Atomically write one or more synchronized camera recordings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = "fast_mail_rgbd_v1"
            handle.attrs["created_unix"] = time.time()
            if metadata is not None:
                handle.attrs["metadata_json"] = json.dumps(metadata)
            observations = handle.require_group("obs")
            lengths = set()
            for camera, recording in recordings.items():
                if len(recording) == 0:
                    raise ValueError(f"recording for {camera} is empty")
                lengths.add(len(recording))
                group = observations.require_group(camera)
                _write_rgbd(group, recording)
                if isinstance(recording, TrackedRGBDRecording):
                    tracking = group.require_group("tracking").require_group("online")
                    tracking.create_dataset(
                        "keypoints_2d", data=_stack(recording.keypoints_2d, "keypoints_2d")
                    )
                    tracking.create_dataset(
                        "keypoints_3d", data=_stack(recording.keypoints_3d, "keypoints_3d")
                    )
                    tracking.create_dataset(
                        "visibility", data=_stack(recording.visibility, "visibility")
                    )
                    tracking.create_dataset(
                        "tracker_frame_index",
                        data=np.asarray(recording.tracker_frame_indices, dtype=np.int64),
                    )
                    tracking.create_dataset("point_ids", data=recording.point_ids)
                    tracking.create_dataset("object_names", data=_string_array(recording.object_names))
                    if recording.selection is not None:
                        tracking.attrs["selection_json"] = json.dumps(recording.selection)
            if len(lengths) != 1:
                raise ValueError(f"camera recordings have different lengths: {sorted(lengths)}")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def next_episode_path(output_dir: str | Path, prefix: str = "episode") -> Path:
    output_dir = Path(output_dir)
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    candidate = output_dir / f"{prefix}_{timestamp}.hdf5"
    suffix = 1
    while candidate.exists():
        candidate = output_dir / f"{prefix}_{timestamp}_{suffix:02d}.hdf5"
        suffix += 1
    return candidate


def write_tracking_group(
    hdf5_path: str | Path,
    method: str,
    camera: str,
    keypoints_2d: np.ndarray,
    keypoints_3d: np.ndarray,
    visibility: np.ndarray,
    frame_indices: np.ndarray,
    point_ids: np.ndarray,
    object_names: np.ndarray,
    selection: Mapping,
    overwrite: bool = False,
):
    """Append online/offline tracking results to an existing RGB-D recording."""
    with h5py.File(hdf5_path, "r+") as handle:
        root = handle.require_group("tracking").require_group(method)
        if camera in root:
            if not overwrite:
                raise FileExistsError(
                    f"tracking/{method}/{camera} already exists; enable overwrite"
                )
            del root[camera]
        group = root.create_group(camera)
        group.create_dataset("keypoints_2d", data=np.asarray(keypoints_2d, np.float32))
        group.create_dataset("keypoints_3d", data=np.asarray(keypoints_3d, np.float32))
        group.create_dataset("visibility", data=np.asarray(visibility, bool))
        group.create_dataset("frame_indices", data=np.asarray(frame_indices, np.int64))
        group.create_dataset("point_ids", data=np.asarray(point_ids))
        group.create_dataset("object_names", data=_string_array(object_names))
        group.attrs["selection_json"] = json.dumps(selection)
