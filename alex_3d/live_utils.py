"""Small shared utilities for interactive RGB-D programs."""

from __future__ import annotations

import json
import re
from pathlib import Path
import time

import cv2
import numpy as np
from alex.track_online_pipe import _color
from alex_3d.selection import collect_direct_points, collect_mask_clicks


class FrequencyGate:
    def __init__(self, frequency_hz: float):
        if frequency_hz <= 0:
            raise ValueError("frequency must be positive")
        self.period = 1.0 / frequency_hz
        self.next_time = None

    def reset(self, now=None):
        self.next_time = (time.monotonic() if now is None else now) + self.period

    def due(self, now=None):
        now = time.monotonic() if now is None else now
        if self.next_time is None:
            self.reset(now)
            return False
        if now < self.next_time:
            return False
        missed = max(1, int((now - self.next_time) // self.period) + 1)
        self.next_time += missed * self.period
        return True


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def output_frame_indices(frame_count, first_frame=15, step=8):
    if first_frame < 0 or step < 1:
        raise ValueError("first_frame must be nonnegative and step must be positive")
    return np.arange(first_frame, frame_count, step, dtype=np.int64)


def selections_for_mode(mode, images, objects, prompt, terminal=None):
    if mode == "prompt":
        return {
            camera: [{"name": name, "text": prompt.get(name, name)} for name in objects]
            for camera in images
        }
    if mode == "mask_click":
        return collect_mask_clicks(images, objects, terminal=terminal)
    if mode == "keypoints":
        return collect_direct_points(images, objects, terminal=terminal)
    raise ValueError(f"unknown selection mode {mode!r}")


def overlay_status(rgb, text, color=(255, 255, 255)):
    display = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(display, (0, 0), (display.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(display, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv2.LINE_AA)
    return display


def render_keypoint_overlay(rgb, xy, visible, point_ids, object_names, show_ids=True):
    """Render tracked points on an RGB frame using stable identity colors."""
    overlay = np.asarray(rgb).copy()
    for point, is_visible, point_id, name in zip(xy, visible, point_ids, object_names):
        if not is_visible or not np.isfinite(point).all():
            continue
        location = tuple(np.rint(point).astype(int))
        color = _color(f"{name}/{point_id}")
        cv2.circle(overlay, location, 4, color, -1, cv2.LINE_AA)
        if show_ids:
            cv2.putText(
                overlay, str(point_id), location,
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
            )
    return overlay


def save_keypoint_overlay(
    output_dir, camera, frame_index, rgb, xy, visible, point_ids, object_names,
):
    """Save one RGB keypoint overlay under ``output_dir/<camera>/``."""
    camera_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(camera))
    directory = Path(output_dir) / camera_name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"frame_{int(frame_index):06d}.png"
    overlay = render_keypoint_overlay(
        rgb, xy, visible, point_ids, object_names, show_ids=True
    )
    if not cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write visualization image: {path}")
    return path


def visualization_fps(timestamps, frame_indices, override=None):
    """Infer playback FPS from selected capture timestamps unless overridden."""
    if override is not None:
        fps = float(override)
        if fps <= 0:
            raise ValueError("visualization_fps must be positive")
        return fps
    selected = np.asarray(timestamps, dtype=np.float64)[np.asarray(frame_indices, dtype=int)]
    if len(selected) < 2:
        return 1.0
    intervals = np.diff(selected)
    intervals = intervals[np.isfinite(intervals) & (intervals > 0)]
    if not len(intervals):
        return 1.0
    return float((len(selected) - 1) / (selected[-1] - selected[0]))


def recording_frequency_hz(handle):
    """Read the nominal capture/tracker input frequency from HDF5 metadata."""
    raw = handle.attrs.get("metadata_json")
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        metadata = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    for key in ("capture_frequency_hz", "tracker_input_frequency_hz"):
        value = metadata.get(key)
        if value is not None and float(value) > 0:
            return float(value)
    return None


def selected_visualization_fps(
    timestamps, frame_indices, output_step, override=None, recording_fps=None,
):
    """Use explicit FPS, then nominal recording FPS, then timestamp fallback."""
    if override is not None:
        return visualization_fps(timestamps, frame_indices, override)
    if recording_fps is not None:
        return float(recording_fps) / int(output_step)
    return visualization_fps(timestamps, frame_indices)


def visualization_video_path(output_dir, method, camera):
    camera_name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(camera))
    return Path(output_dir) / f"{method}_{camera_name}.mp4"


def write_keypoint_video(
    path, rgb_frames, frame_indices, tracks, visibility, point_ids, object_names, fps,
):
    """Write timestamp-paced RGB keypoint overlays to an MP4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(rgb_frames[int(frame_indices[0])])
    height, width = first.shape[:2]
    encoded_size = (width + width % 2, height + height % 2)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), encoded_size
    )
    if not writer.isOpened():
        raise OSError(f"Could not open visualization video: {path}")
    try:
        for index, xy, visible in zip(frame_indices, tracks, visibility):
            overlay = render_keypoint_overlay(
                rgb_frames[int(index)], xy, visible, point_ids, object_names,
                show_ids=True,
            )
            overlay = cv2.copyMakeBorder(
                overlay, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT
            )
            writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return path


def play_keypoint_video_frames(
    window_name, rgb_frames, frame_indices, tracks, visibility,
    point_ids, object_names, fps,
):
    """Display selected overlays at their intended playback frequency."""
    delay_ms = max(1, int(round(1000.0 / fps)))
    for index, xy, visible in zip(frame_indices, tracks, visibility):
        overlay = render_keypoint_overlay(
            rgb_frames[int(index)], xy, visible, point_ids, object_names,
            show_ids=True,
        )
        cv2.imshow(window_name, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(delay_ms) & 0xFF in (27, ord("q")):
            break
    cv2.destroyWindow(window_name)
