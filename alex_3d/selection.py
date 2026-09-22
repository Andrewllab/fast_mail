"""OpenCV selection tools for SAM masks and direct CoTracker queries."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Mapping, Sequence

import cv2
import numpy as np

from alex.track_online_pipe import direct_point_queries
from alex_3d.terminal_input import TerminalKeyReader


def collect_mask_clicks(
    images: Mapping[str, np.ndarray],
    object_names: Sequence[str],
    terminal: TerminalKeyReader | None = None,
):
    """Click in the image; press Enter or q in the launching terminal."""
    selections = {}
    context = nullcontext(terminal) if terminal is not None else TerminalKeyReader()
    with context as keys:
        for camera, frame in images.items():
            selections[camera] = []
            for name in object_names:
                points, labels = [], []
                display = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
                window = f"SAM3 mask: {camera}/{name}"

                def mouse(event, x, y, flags, userdata):
                    if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
                        return
                    label = int(event == cv2.EVENT_LBUTTONDOWN)
                    points.append([float(x), float(y)])
                    labels.append(label)
                    cv2.drawMarker(display, (x, y), (0, 220, 0) if label else (0, 0, 220),
                                   cv2.MARKER_CROSS, 11, 2)

                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(window, mouse)
                print(
                    f"Select mask {camera}/{name}: left-click foreground, right-click "
                    "background; press Enter in this terminal to confirm or q to cancel.",
                    flush=True,
                )
                try:
                    while True:
                        cv2.imshow(window, display)
                        cv2.waitKey(20)
                        key = keys.poll()
                        if key == "enter":
                            if 1 in labels:
                                break
                            print("Add at least one foreground click before confirming.", flush=True)
                        if key == "q":
                            raise RuntimeError("mask selection cancelled")
                finally:
                    cv2.destroyWindow(window)
                selections[camera].append({"name": name, "points": points, "labels": labels})
    return selections


def collect_direct_points(
    images: Mapping[str, np.ndarray],
    object_names: Sequence[str],
    terminal: TerminalKeyReader | None = None,
):
    """Left-click queries; press Enter or q in the launching terminal."""
    selections = {}
    context = nullcontext(terminal) if terminal is not None else TerminalKeyReader()
    with context as keys:
        for camera, frame in images.items():
            selections[camera] = []
            for name in object_names:
                points = []
                display = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
                window = f"CoTracker points: {camera}/{name}"

                def mouse(event, x, y, flags, userdata):
                    if event != cv2.EVENT_LBUTTONDOWN:
                        return
                    points.append([float(x), float(y)])
                    cv2.drawMarker(display, (x, y), (0, 220, 0), cv2.MARKER_CROSS, 11, 2)
                    cv2.putText(display, str(len(points)), (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)

                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(window, mouse)
                print(
                    f"Select keypoints for {camera}/{name}: left-click points in the image; "
                    "press Enter in this terminal to confirm or q to cancel.",
                    flush=True,
                )
                try:
                    while True:
                        cv2.imshow(window, display)
                        cv2.waitKey(20)
                        key = keys.poll()
                        if key == "enter":
                            if points:
                                break
                            print("Click at least one keypoint before confirming.", flush=True)
                        if key == "q":
                            raise RuntimeError("keypoint selection cancelled")
                finally:
                    cv2.destroyWindow(window)
                selections[camera].append({"name": name, "points": points})
    return selections


def flatten_direct_points(images, selections, camera_names):
    return direct_point_queries(images, selections, camera_names)


def initialize_tracker_from_points(tracker, images, selections):
    """Initialize either alex or alex_batch online tracker with exact points."""
    return tracker.initialize_points(images, selections)


class DirectPointSegmenter:
    """Placeholder segmenter for a pipeline initialized with exact points."""

    def __init__(self, camera_names):
        self.camera_names = tuple(camera_names)

    def release_models(self):
        pass


def sample_mask_grid(masks, shape, grid_spacing=16, max_points_per_object=128):
    """Return exact ``(x,y)`` queries and object names from mask-grid sampling."""
    if grid_spacing < 1:
        raise ValueError("grid_spacing must be positive")
    rows = np.arange(grid_spacing // 2, shape[0], grid_spacing)
    columns = np.arange(grid_spacing // 2, shape[1], grid_spacing)
    grid_x, grid_y = np.meshgrid(columns, rows)
    grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    groups, names = [], []
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != shape:
            raise ValueError(f"mask {name} has shape {mask.shape}, expected {shape}")
        points = grid[mask[grid[:, 1], grid[:, 0]]]
        if not len(points):
            raise ValueError(f"no grid point inside {name}; reduce grid spacing")
        if max_points_per_object and len(points) > max_points_per_object:
            points = points[
                np.linspace(0, len(points) - 1, max_points_per_object, dtype=int)
            ]
        groups.append(points)
        names.extend([name] * len(points))
    return np.concatenate(groups).astype(np.float32), np.asarray(names)
