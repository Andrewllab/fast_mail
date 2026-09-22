"""First-frame SAM3 masks and causal, persistent multi-camera CoTracker3 tracks.

Images are RGB uint8 HWC arrays; coordinates are (x, y) pixels in those images.
Online inference uses 16-frame windows with eight-frame overlap and fixed IDs.
"""

from __future__ import annotations

import colorsys
import copy
import hashlib
import re
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import cv2
import numpy as np
import torch


def validate_images(images: Mapping[str, np.ndarray], camera_names: Sequence[str]):
    if set(images) != set(camera_names):
        raise ValueError(f"Expected cameras {list(camera_names)}, got {list(images)}")
    for camera, frame in images.items():
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"{camera}: expected RGB uint8 [H,W,3], got {frame.shape}/{frame.dtype}")
        if min(frame.shape[:2]) < 2:
            raise ValueError(f"{camera}: image dimensions must be at least two pixels")


def direct_point_queries(images, selections, camera_names):
    """Validate named direct-click selections and return points/names per camera."""
    validate_images(images, camera_names)
    if set(selections) != set(camera_names):
        raise ValueError("Direct selections must contain exactly the configured cameras")
    result = {}
    for camera in camera_names:
        points, names = [], []
        height, width = images[camera].shape[:2]
        for selection in selections[camera]:
            name = str(selection["name"])
            selected = np.asarray(selection["points"], dtype=np.float32)
            if selected.ndim != 2 or selected.shape[1] != 2 or not len(selected):
                raise ValueError(f"Invalid direct points for {camera}/{name}")
            if not np.isfinite(selected).all() or (selected < 0).any():
                raise ValueError(f"Invalid direct point for {camera}/{name}")
            if (selected >= [width, height]).any():
                raise ValueError(f"Direct point outside image for {camera}/{name}")
            points.append(selected)
            names.extend([name] * len(selected))
        if not points:
            raise ValueError(f"At least one named direct-point selection is required for {camera}")
        result[camera] = (np.concatenate(points), np.asarray(names))
    return result


def _camera_names(names: Sequence[str]) -> tuple[str, ...]:
    if (
        isinstance(names, str)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("camera_names must be a nonempty sequence of unique names")
    return tuple(names)


def _color(identity: str) -> tuple[int, int, int]:
    hue = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) / 2**32
    return tuple(int(channel * 255) for channel in colorsys.hsv_to_rgb(hue, 0.8, 1.0))


def _safe_name(name: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    return f"{readable}_{hashlib.sha256(name.encode()).hexdigest()[:6]}"


def save_rgb(path: str | Path, frame: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
        raise OSError(f"Could not write image: {path}")
    return path


def _show_images(images: Mapping[str, np.ndarray], title: str):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(images), squeeze=False)
    for axis, (camera, frame) in zip(axes.flat, images.items()):
        axis.imshow(frame)
        axis.set_title(camera)
        axis.axis("off")
    figure.suptitle(title)
    plt.show()


class SAM3Segmenter:
    """Segment initial RGB images using per-camera text or foreground/background clicks.

    Selection format: {camera: [{"name": "cup", "text": "red cup"}]} or
    {camera: [{"name": "cup", "points": [[120, 90]], "labels": [1]}]}.
    Text selection defaults to the best instance; instances="all" keeps separate
    masks named name:0, name:1, ... in descending detection-score order.
    Object labels across cameras express user intent, not visual correspondence.
    """

    def __init__(
        self,
        camera_names: Sequence[str],
        device: str = "auto",
        model_id: str = "facebook/sam3",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        threshold: float = 0.5,
    ):
        self.camera_names = _camera_names(camera_names)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.model_id = model_id
        self.load_options = {"cache_dir": cache_dir, "local_files_only": local_files_only}
        if not 0 <= threshold <= 1:
            raise ValueError("Segmentation threshold must lie in [0,1]")
        self.threshold = threshold
        self._models = {}
        self.masks = None
        self.initial_images = None

    def _load(self, kind: str):
        if kind in self._models:
            return self._models[kind]

        from transformers import (
            Sam3Model,
            Sam3Processor,
            Sam3TrackerModel,
            Sam3TrackerProcessor,
        )

        model_class, processor_class = (
            (Sam3Model, Sam3Processor)
            if kind == "text"
            else (Sam3TrackerModel, Sam3TrackerProcessor)
        )
        model = model_class.from_pretrained(
            self.model_id, **self.load_options
        ).to(self.device).eval()
        processor = processor_class.from_pretrained(
            self.model_id, **self.load_options
        )
        self._models[kind] = (model, processor)
        return self._models[kind]

    @torch.inference_mode()
    def segment(self, images: Mapping[str, np.ndarray], selections: Mapping[str, list[dict]]):
        validate_images(images, self.camera_names)
        if set(selections) != set(self.camera_names):
            raise ValueError("Provide an object-selection list for every camera")
        masks_by_camera = {}
        for camera in self.camera_names:
            frame = images[camera]
            object_masks = {}
            for selection in selections[camera]:
                name = str(selection["name"])
                if ("text" in selection) == ("points" in selection):
                    raise ValueError(f"{camera}/{name}: provide exactly one of text or points")
                if "text" in selection:
                    model, processor = self._load("text")
                    inputs = processor(images=frame, text=selection["text"], return_tensors="pt").to(self.device)
                    outputs = model(**inputs)
                    result = processor.post_process_instance_segmentation(
                        outputs, threshold=self.threshold, mask_threshold=0.5,
                        target_sizes=inputs["original_sizes"].tolist(),
                    )[0]
                    order = result["scores"].argsort(descending=True)
                    instances = selection.get("instances", "best")
                    if instances not in ("best", "all"):
                        raise ValueError("instances must be 'best' or 'all'")
                    if instances == "best":
                        order = order[:1]
                    candidates = {
                        name if instances == "best" else f"{name}:{rank}": result["masks"][index].cpu().numpy().astype(bool)
                        for rank, index in enumerate(order)
                    }
                    if not candidates:
                        raise ValueError(f"SAM3 found no instance for {camera}/{name}; change prompt or use clicks")
                else:
                    points = np.asarray(selection["points"], dtype=np.float32)
                    labels = np.asarray(selection.get("labels", [1] * len(points)))
                    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
                        raise ValueError("Click points must be nonempty [N,2] pixel coordinates")
                    if labels.shape != (len(points),) or not np.isin(labels, [0, 1]).all() or not (labels == 1).any():
                        raise ValueError("Click labels must be 0/1 and include a foreground click")
                    if not np.isfinite(points).all() or (points < 0).any() or (points >= [frame.shape[1], frame.shape[0]]).any():
                        raise ValueError(f"{camera}/{name}: clicks are outside the processed image")
                    model, processor = self._load("click")
                    inputs = processor(
                        images=frame, input_points=[[points.tolist()]],
                        input_labels=[[labels.tolist()]], return_tensors="pt",
                    ).to(self.device)
                    outputs = model(**inputs, multimask_output=True)
                    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
                    best = int(outputs.iou_scores[0, 0].argmax())
                    candidates = {name: masks[0, best].numpy().astype(bool)}
                for object_name, mask in candidates.items():
                    if object_name in object_masks:
                        raise ValueError(f"Duplicate object name: {camera}/{object_name}")
                    if mask.shape != frame.shape[:2] or not mask.any():
                        raise ValueError(f"Empty or incorrectly sized mask: {camera}/{object_name}")
                    object_masks[object_name] = mask
            if not object_masks:
                raise ValueError(f"No objects selected for {camera}; remove this camera or select an object")
            masks_by_camera[camera] = object_masks
        self.initial_images = {camera: frame.copy() for camera, frame in images.items()}
        self.masks = masks_by_camera
        return masks_by_camera

    @staticmethod
    def collect_clicks(images: Mapping[str, np.ndarray], object_names: Sequence[str]):
        """Open one selection window per camera/object. Enter accepts; right click excludes."""
        import matplotlib.pyplot as plt

        selections = {}
        for camera, frame in images.items():
            selections[camera] = []
            for name in object_names:
                points, labels = [], []
                figure, axis = plt.subplots()
                axis.imshow(frame)
                axis.set_title(f"{camera}: {name}\nLeft: foreground; right: background; Enter: accept")

                def on_click(event):
                    if event.inaxes == axis and event.button in (1, 3):
                        points.append([float(event.xdata), float(event.ydata)])
                        labels.append(1 if event.button == 1 else 0)
                        axis.plot(event.xdata, event.ydata, "+", color="lime" if labels[-1] else "red")
                        figure.canvas.draw_idle()

                def on_key(event):
                    if event.key == "enter":
                        plt.close(figure)

                figure.canvas.mpl_connect("button_press_event", on_click)
                figure.canvas.mpl_connect("key_press_event", on_key)
                plt.show()
                if 1 not in labels:
                    raise ValueError(f"No foreground click selected for {camera}/{name}")
                selections[camera].append({"name": name, "points": points, "labels": labels})
        return selections

    def visualize(self, output_dir: str | Path | None = None, show: bool = False):
        if self.masks is None:
            raise RuntimeError("Call segment() before visualization")
        overlays = {}
        for camera, masks in self.masks.items():
            overlay = self.initial_images[camera].copy()
            for index, (name, mask) in enumerate(masks.items()):
                color = _color(name)
                overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array(color)).astype(np.uint8)
                cv2.putText(overlay, name, (5, 18 + index * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            overlays[camera] = overlay
            if output_dir is not None:
                save_rgb(Path(output_dir) / f"{_safe_name(camera)}_segmentation.png", overlay)
                np.savez_compressed(
                    Path(output_dir) / f"{_safe_name(camera)}_masks.npz",
                    masks=np.stack(list(masks.values())), object_names=np.array(list(masks)),
                )
        if show:
            _show_images(overlays, "Initial object masks")
        return overlays

    def release_models(self):
        """Free segmentation weights once initialization is complete."""
        self._models.clear()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


@dataclass
class TrackFrame:
    """One causal result. IDs remain fixed within a camera until reset()."""

    frame_index: int
    xy: np.ndarray
    visible: np.ndarray
    point_ids: np.ndarray
    object_names: np.ndarray
    status: str


class OnlineKeypointTracker:
    """Buffer 16 frames initially, then infer every eight newly received frames.

    Each camera owns an independent predictor with persistent query IDs. The
    getter returns None during warmup, then the last inferred frame, which may
    precede the latest received frame. Unfinished tails are not inferred.
    """

    window_size = 16
    step = 8

    def __init__(
        self,
        camera_names: Sequence[str],
        device: str = "auto",
        grid_spacing: int = 16,
        max_points_per_object: int | None = 128,
        support_grid: bool = True,
        hub_repo: str = "facebookresearch/co-tracker",
        hub_source: str = "github",
        predictor_factory: Callable | None = None,
    ):
        self.camera_names = _camera_names(camera_names)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        if grid_spacing < 1 or (max_points_per_object is not None and max_points_per_object < 1):
            raise ValueError("Grid spacing and point cap must be positive")
        self.grid_spacing = grid_spacing
        self.max_points_per_object = max_points_per_object
        self.support_grid = support_grid
        self.predictor_factory = predictor_factory or (
            lambda: torch.hub.load(hub_repo, "cotracker3_online", source=hub_source, trust_repo=True)
        )
        self.reset()

    def reset(self):
        """Discard the episode, models and IDs; initialize again for a new stream."""
        self.predictors = {}
        self.buffers = {}
        self.identities = {}
        self.shapes = {}
        self.latest = None
        self.tracks = {}
        self.visibility = {}
        self.updated = False
        self.inference_count = 0
        self.last_inference_ms = {}
        self.inference_times_ms = []
        self.frame_index = -1
        self.closed = False

    def _sample(self, masks: Mapping[str, np.ndarray], shape: tuple[int, int]):
        rows = np.arange(self.grid_spacing // 2, shape[0], self.grid_spacing)
        columns = np.arange(self.grid_spacing // 2, shape[1], self.grid_spacing)
        grid_x, grid_y = np.meshgrid(columns, rows)
        grid = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        groups, names = [], []
        for name, mask in masks.items():
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != shape:
                raise ValueError(f"Mask {name} has shape {mask.shape}, expected {shape}")
            points = grid[mask[grid[:, 1], grid[:, 0]]]
            if not len(points):
                raise ValueError(f"No grid point inside {name}; reduce grid_spacing or refine the mask")
            if self.max_points_per_object is not None and len(points) > self.max_points_per_object:
                points = points[np.linspace(0, len(points) - 1, self.max_points_per_object, dtype=int)]
            groups.append(points)
            names.extend([name] * len(points))
        if not groups:
            raise ValueError("At least one object mask is required per camera")
        return np.concatenate(groups).astype(np.float32), np.array(names)

    def _video(self, frames):
        return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0).to(self.device, dtype=torch.float32)

    @torch.inference_mode()
    def initialize(self, images: Mapping[str, np.ndarray], masks: Mapping[str, Mapping[str, np.ndarray]]):
        if self.frame_index >= 0:
            raise RuntimeError("Already initialized; call reset() for a new stream")
        validate_images(images, self.camera_names)
        if set(masks) != set(self.camera_names):
            raise ValueError("Mask camera names must match image camera names")
        sampled = {camera: self._sample(masks[camera], images[camera].shape[:2]) for camera in self.camera_names}
        for camera in self.camera_names:
            points, names = sampled[camera]
            predictor = self.predictor_factory().to(self.device).eval()
            if any(predictor is other for other in self.predictors.values()):
                raise ValueError("predictor_factory must return a separate predictor per camera")
            queries = torch.from_numpy(np.column_stack([np.zeros(len(points)), points])).float().unsqueeze(0).to(self.device)
            predictor(self._video([images[camera]]), is_first_step=True, queries=queries, add_support_grid=self.support_grid)
            if predictor.step != self.step or predictor.model.window_len != self.window_size or getattr(predictor, "v2", False):
                raise RuntimeError("Expected CoTracker3 online with window_size=16 and step=8")
            self.predictors[camera] = predictor
            self.buffers[camera] = deque([images[camera].copy()])
            self.identities[camera] = (np.arange(len(points)), names)
            self.shapes[camera] = images[camera].shape
        self.frame_index = 0
        return None

    @torch.inference_mode()
    def initialize_points(self, images: Mapping[str, np.ndarray], selections):
        """Initialize directly from named pixel queries without running SAM."""
        if self.frame_index >= 0:
            raise RuntimeError("Already initialized; call reset() for a new stream")
        selected = direct_point_queries(images, selections, self.camera_names)
        for camera in self.camera_names:
            points, names = selected[camera]
            predictor = self.predictor_factory().to(self.device).eval()
            if any(predictor is other for other in self.predictors.values()):
                raise ValueError("predictor_factory must return a separate predictor per camera")
            queries = torch.from_numpy(
                np.column_stack([np.zeros(len(points), dtype=np.float32), points])
            ).unsqueeze(0).to(self.device)
            predictor(
                self._video([images[camera]]), is_first_step=True,
                queries=queries, add_support_grid=self.support_grid,
            )
            if (predictor.step != self.step or predictor.model.window_len != self.window_size
                    or getattr(predictor, "v2", False)):
                raise RuntimeError("Expected CoTracker3 online with window_size=16 and step=8")
            self.predictors[camera] = predictor
            self.buffers[camera] = deque([images[camera].copy()])
            self.identities[camera] = (np.arange(len(points)), names)
            self.shapes[camera] = images[camera].shape
        self.frame_index = 0
        return None

    @torch.inference_mode()
    def push(self, images: Mapping[str, np.ndarray]):
        if self.frame_index < 0 or self.closed:
            raise RuntimeError("Initialize an open stream before push()")
        validate_images(images, self.camera_names)
        for camera, frame in images.items():
            if frame.shape != self.shapes[camera]:
                raise ValueError(f"Resolution changed for {camera}; reset and reinitialize")
        self.updated = False
        self.last_inference_ms = {}
        self.frame_index += 1
        for camera in self.camera_names:
            self.buffers[camera].append(images[camera].copy())
        if len(self.buffers[self.camera_names[0]]) < self.window_size:
            return self.get_latest_keypoints()
        if self.inference_count == 0:
            print("Starting CoTracker3 online inference: 16 frames accumulated.", flush=True)
        results = {}
        for camera in self.camera_names:
            predictor = self.predictors[camera]
            buffer = self.buffers[camera]
            try:
                started = time.perf_counter()
                tracks, visibility = predictor(self._video(buffer), is_first_step=False, add_support_grid=self.support_grid)
                self.last_inference_ms[camera] = (time.perf_counter() - started) * 1000
                if tracks is None or tracks.shape[1] != self.frame_index + 1:
                    raise RuntimeError("CoTracker returned an unexpected timeline; online API may have changed")
                points = tracks[0].detach().cpu().numpy().copy()
                visible = visibility[0].detach().cpu().numpy().astype(bool)
            except Exception:
                self.closed = True
                raise
            for _ in range(self.step):
                buffer.popleft()
            point_ids, names = self.identities[camera]
            if points.shape[1:] != (len(point_ids), 2):
                self.closed = True
                raise RuntimeError("CoTracker changed the number of query points")
            height, width = self.shapes[camera][:2]
            visible &= np.isfinite(points).all(axis=-1) & (points >= 0).all(axis=-1) & (points < [width, height]).all(axis=-1)
            self.tracks[camera] = points
            self.visibility[camera] = visible
            results[camera] = TrackFrame(self.frame_index, points[-1].copy(), visible[-1].copy(), point_ids.copy(), names.copy(), "inferred")
        self.inference_count += 1
        self.inference_times_ms.append(dict(self.last_inference_ms))
        self.updated = True
        self.latest = results
        return self.get_latest_keypoints()

    def get_latest_keypoints(self, camera: str | None = None):
        """Return only the last inferred frame (or None before 16 input frames)."""
        if camera is not None and camera not in self.camera_names:
            raise KeyError(camera)
        if self.latest is None:
            return None
        return copy.deepcopy(self.latest if camera is None else self.latest[camera])

    def finish(self):
        """Close without inferring an incomplete window; retain the last result."""
        self.closed = True
        return self.get_latest_keypoints()

    @staticmethod
    def visualize(images: Mapping[str, np.ndarray], results: Mapping[str, TrackFrame], show_ids: bool = False):
        overlays = {}
        for camera, result in results.items():
            overlay = images[camera].copy()
            for point, visible, identity, name in zip(result.xy, result.visible, result.point_ids, result.object_names):
                if visible:
                    location = tuple(np.round(point).astype(int))
                    color = _color(f"{name}/{identity}")
                    cv2.circle(overlay, location, 3, color, -1, cv2.LINE_AA)
                    if show_ids:
                        cv2.putText(overlay, str(identity), location, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cv2.putText(overlay, f"{camera} frame {result.frame_index} {result.status}", (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            overlays[camera] = overlay
        return overlays


class OnlineTrackingPipeline:
    """Orchestrate segmentation, tracking, timing and optional episode recording.

    initialize() consumes frame zero; push() consumes each subsequent synchronized
    camera dictionary exactly once. Getters describe the last inferred frame.
    keep_history=False avoids retaining RGB/CPU trajectory history for live use.
    CoTracker itself still retains its growing internal prediction history.
    """

    def __init__(self, segmenter: SAM3Segmenter, tracker: OnlineKeypointTracker, keep_history: bool = True):
        if segmenter.camera_names != tracker.camera_names:
            raise ValueError("Segmenter and tracker camera order must agree")
        self.segmenter = segmenter
        self.tracker = tracker
        self.keep_history = keep_history
        self.history = {camera: [] for camera in tracker.camera_names}
        self.frames = {camera: [] for camera in tracker.camera_names}
        self.inference_times_ms = []
        self.latest_images = None
        self.received_images = None
        self.latest = None
    def _record(self, images, result):
        self.received_images = {camera: frame.copy() for camera, frame in images.items()}
        if self.tracker.updated:
            self.latest_images = self.received_images
            self.inference_times_ms.append(sum(self.tracker.last_inference_ms.values()))
        self.latest = result
        if self.keep_history:
            for camera in self.tracker.camera_names:
                self.frames[camera].append(images[camera].copy())
                if self.tracker.updated:
                    point_ids, names = self.tracker.identities[camera]
                    self.history[camera] = [
                        TrackFrame(index, points.copy(), visible.copy(), point_ids.copy(), names.copy(), "inferred")
                        for index, (points, visible) in enumerate(zip(self.tracker.tracks[camera], self.tracker.visibility[camera]))
                    ]
        return result

    def initialize(self, images, selections, release_segmenter: bool = True):
        if self.tracker.frame_index >= 0:
            raise RuntimeError("Create a new pipeline for a new episode")
        masks = self.segmenter.segment(images, selections)
        if release_segmenter:
            self.segmenter.release_models()
        result = self.tracker.initialize(images, masks)
        return self._record(images, result)

    def initialize_points(self, images, selections):
        """Initialize from named direct-click pixels and bypass segmentation."""
        if self.tracker.frame_index >= 0:
            raise RuntimeError("Create a new pipeline for a new episode")
        result = self.tracker.initialize_points(images, selections)
        self.direct_point_selections = copy.deepcopy(selections)
        return self._record(images, result)

    def push(self, images):
        result = self.tracker.push(images)
        return self._record(images, result)

    def get_latest_keypoints(self, camera: str | None = None):
        return self.tracker.get_latest_keypoints(camera)

    def visualize_current(self, output_dir: str | Path | None = None, show: bool = False, show_ids: bool = False):
        if self.latest is None:
            if self.received_images is None:
                raise RuntimeError("Initialize the pipeline first")
            overlays = {camera: frame.copy() for camera, frame in self.received_images.items()}
        else:
            overlays = self.tracker.visualize(self.latest_images, self.latest, show_ids=show_ids)
        if output_dir is not None:
            for camera, overlay in overlays.items():
                save_rgb(Path(output_dir) / f"{_safe_name(camera)}_current.png", overlay)
        if show:
            _show_images(overlays, "Current keypoints")
        return overlays

    def export(self, output_dir: str | Path, fps: float = 15.0, save_videos: bool = True, trail_length: int = 20):
        """Export the latest inferred timeline, including overlap refinements.

        Unprocessed tail frames are excluded; no stale positions are substituted.
        """
        if not self.keep_history or self.latest is None:
            raise RuntimeError("Export requires keep_history=True and at least 16 received frames")
        if fps <= 0 or trail_length < 0:
            raise ValueError("fps must be positive and trail_length nonnegative")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {}
        for camera, history in self.history.items():
            stem = _safe_name(camera)
            tracks = np.stack([result.xy for result in history])
            visible = np.stack([result.visible for result in history])
            npz_path = output_dir / f"{stem}_tracks.npz"
            np.savez_compressed(
                npz_path, tracks=tracks, visibility=visible,
                point_ids=history[0].point_ids, object_names=history[0].object_names,
                frame_indices=np.array([result.frame_index for result in history]),
                inference_ms=np.array(self.inference_times_ms[:len(history)]),
                status=np.array([result.status for result in history]),
                image_size_wh=np.array(self.frames[camera][0].shape[1::-1]),
            )
            paths[camera] = {"tracks": str(npz_path)}
            if not save_videos:
                continue
            video_path = output_dir / f"{stem}_tracks.mp4"
            height, width = self.frames[camera][0].shape[:2]
            encoded_size = (width + width % 2, height + height % 2)
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, encoded_size)
            if not writer.isOpened():
                raise OSError(f"Video encoder could not open {video_path}")
            try:
                for frame_index, (frame, result) in enumerate(zip(self.frames[camera], history)):
                    overlay = frame.copy()
                    for point_index, name in enumerate(result.object_names):
                        color = _color(f"{name}/{result.point_ids[point_index]}")
                        for previous in range(max(1, frame_index - trail_length + 1), frame_index + 1):
                            if visible[previous - 1, point_index] and visible[previous, point_index]:
                                start = tuple(np.round(tracks[previous - 1, point_index]).astype(int))
                                end = tuple(np.round(tracks[previous, point_index]).astype(int))
                                cv2.line(overlay, start, end, color, 1, cv2.LINE_AA)
                    overlay = self.tracker.visualize({camera: overlay}, {camera: result})[camera]
                    overlay = cv2.copyMakeBorder(overlay, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT)
                    writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            paths[camera]["video"] = str(video_path)
        return paths

    def finish(self):
        return self.tracker.finish()

    @staticmethod
    def play_video(path: str | Path):
        """Play an exported video in a GUI window; Escape or q closes it."""
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise OSError(f"Cannot open {path}")
        delay = max(1, round(1000 / max(capture.get(cv2.CAP_PROP_FPS), 1)))
        try:
            while True:
                available, frame = capture.read()
                if not available:
                    break
                cv2.imshow(str(path), frame)
                if cv2.waitKey(delay) & 0xFF in (27, ord("q")):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()
