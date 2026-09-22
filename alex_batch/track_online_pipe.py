"""Batched multi-camera SAM3 segmentation and CoTracker3 online tracking.

This module keeps the public classes of ``alex.track_online_pipe`` while using
one SAM3 model call per selection mode and one CoTracker3 predictor for all
synchronized camera views.
"""

from __future__ import annotations

import copy
import time
import types
from collections import deque
from typing import Mapping, Sequence, Callable

import numpy as np
import torch

from alex.track_online_pipe import (
    SAM3Segmenter as _SAM3Segmenter,
    OnlineKeypointTracker as _OnlineKeypointTracker,
    OnlineTrackingPipeline as _OnlineTrackingPipeline,
    TrackFrame,
    _camera_names,
    direct_point_queries,
    validate_images,
)


class SAM3Segmenter(_SAM3Segmenter):
    """Batch SAM3 text and click segmentation across camera views."""

    def segment(self, images: Mapping[str, np.ndarray], selections: Mapping[str, list[dict]]):
        validate_images(images, self.camera_names)
        if set(selections) != set(self.camera_names):
            raise ValueError("Provide an object-selection list for every camera")

        masks_by_camera = {camera: {} for camera in self.camera_names}
        text_jobs, click_jobs = [], []
        for camera in self.camera_names:
            for selection in selections[camera]:
                name = str(selection["name"])
                has_text, has_points = "text" in selection, "points" in selection
                if has_text == has_points:
                    raise ValueError(f"{camera}/{name}: provide exactly one of text or points")
                frame = images[camera]
                if has_text:
                    instances = selection.get("instances", "best")
                    if instances not in ("best", "all"):
                        raise ValueError("instances must be 'best' or 'all'")
                    text_jobs.append((camera, name, str(selection["text"]), instances))
                    continue
                points = np.asarray(selection["points"], dtype=np.float32)
                labels = np.asarray(selection.get("labels", [1] * len(points)), dtype=np.int64)
                if points.ndim != 2 or points.shape[1] != 2 or not len(points):
                    raise ValueError("Click points must be nonempty [N,2] pixel coordinates")
                if labels.shape != (len(points),) or not np.isin(labels, [0, 1]).all() or not (labels == 1).any():
                    raise ValueError("Click labels must be 0/1 and include a foreground click")
                if not np.isfinite(points).all() or (points < 0).any() or (points >= [frame.shape[1], frame.shape[0]]).any():
                    raise ValueError(f"{camera}/{name}: clicks are outside the processed image")
                click_jobs.append((camera, name, points, labels))

        if text_jobs:
            model, processor = self._load("text")
            inputs = processor(
                images=[images[camera] for camera, *_ in text_jobs],
                text=[text for _, _, text, _ in text_jobs],
                return_tensors="pt",
            ).to(self.device)
            outputs = model(**inputs)
            results = processor.post_process_instance_segmentation(
                outputs, threshold=self.threshold, mask_threshold=0.5,
                target_sizes=inputs["original_sizes"].tolist(),
            )
            for job, result in zip(text_jobs, results):
                camera, name, _, instances = job
                order = result["scores"].argsort(descending=True)
                if instances == "best":
                    order = order[:1]
                if not len(order):
                    raise ValueError(f"SAM3 found no instance for {camera}/{name}")
                for rank, index in enumerate(order):
                    object_name = name if instances == "best" else f"{name}:{rank}"
                    self._store_mask(masks_by_camera[camera], object_name,
                                     result["masks"][index].detach().cpu().numpy(), images[camera])

        if click_jobs:
            model, processor = self._load("click")
            max_points = max(len(points) for _, _, points, _ in click_jobs)
            point_batch, label_batch = [], []
            for _, _, points, labels in click_jobs:
                pad = max_points - len(points)
                point_batch.append([np.pad(points, ((0, pad), (0, 0))).tolist()])
                label_batch.append([np.pad(labels, (0, pad), constant_values=-1).tolist()])
            inputs = processor(
                images=[images[camera] for camera, *_ in click_jobs],
                input_points=point_batch, input_labels=label_batch,
                return_tensors="pt",
            ).to(self.device)
            outputs = model(**inputs, multimask_output=True)
            processed = processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
            )
            for batch_index, (camera, name, _, _) in enumerate(click_jobs):
                masks = processed[batch_index]
                best = int(outputs.iou_scores[batch_index, 0].argmax())
                if masks.ndim == 4:
                    mask = masks[0, best]
                else:
                    mask = masks[best]
                self._store_mask(masks_by_camera[camera], name, mask.numpy(), images[camera])

        for camera in self.camera_names:
            if not masks_by_camera[camera]:
                raise ValueError(f"No objects selected for {camera}")
        self.initial_images = {camera: frame.copy() for camera, frame in images.items()}
        self.masks = masks_by_camera
        return masks_by_camera

    @staticmethod
    def _store_mask(target, name, mask, frame):
        mask = np.asarray(mask, dtype=bool)
        if name in target:
            raise ValueError(f"Duplicate object name: {name}")
        if mask.shape != frame.shape[:2] or not mask.any():
            raise ValueError(f"Empty or incorrectly sized mask for {name}")
        target[name] = mask


class OnlineKeypointTracker(_OnlineKeypointTracker):
    """One CoTracker3 online predictor with a batch entry per camera."""

    def reset(self):
        self.predictor = None
        self.predictors = {}
        self.buffers = {}
        self.identities = {}
        self.point_counts = {}
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

    def _batch_video(self):
        height = max(self.shapes[camera][0] for camera in self.camera_names)
        width = max(self.shapes[camera][1] for camera in self.camera_names)
        batch = []
        for camera in self.camera_names:
            camera_frames = []
            for frame in self.buffers[camera]:
                padded = np.zeros((height, width, 3), dtype=np.uint8)
                padded[:frame.shape[0], :frame.shape[1]] = frame
                camera_frames.append(padded)
            batch.append(np.stack(camera_frames))
        return torch.from_numpy(np.stack(batch)).permute(0, 1, 4, 2, 3).to(
            self.device, dtype=torch.float32
        )

    def _append_support_grid(self, queries, height, width):
        if not self.support_grid:
            return queries
        grid_size = int(getattr(self.predictor, "support_grid_size", 6))
        y = np.linspace(0, max(height - 1, 0), grid_size, dtype=np.float32)
        x = np.linspace(0, max(width - 1, 0), grid_size, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(x, y)
        grid = np.stack([np.zeros(grid_x.size, dtype=np.float32),
                         grid_x.ravel(), grid_y.ravel()], axis=-1)
        tiled = np.broadcast_to(grid[None], (len(self.camera_names), *grid.shape)).copy()
        return np.concatenate([queries, tiled], axis=1)

    def _fix_batched_model_stride(self):
        model = getattr(self.predictor, "model", None)
        if model is None or not hasattr(model, "forward_window"):
            return
        original = model.forward_window
        if getattr(original, "_alex_batch_contiguous", False):
            return

        def contiguous_forward_window(instance, *args, **kwargs):
            if "coords" in kwargs:
                kwargs["coords"] = kwargs["coords"].contiguous()
            elif len(args) >= 2:
                args = list(args)
                args[1] = args[1].contiguous()
            return original(*args, **kwargs)

        contiguous_forward_window._alex_batch_contiguous = True
        model.forward_window = types.MethodType(contiguous_forward_window, model)

    @torch.inference_mode()
    def initialize(self, images: Mapping[str, np.ndarray], masks: Mapping[str, Mapping[str, np.ndarray]]):
        if self.frame_index >= 0:
            raise RuntimeError("Already initialized; call reset() for a new stream")
        validate_images(images, self.camera_names)
        if set(masks) != set(self.camera_names):
            raise ValueError("Mask camera names must match image camera names")
        sampled = {}
        for camera in self.camera_names:
            self.shapes[camera] = images[camera].shape
            sampled[camera] = self._sample(masks[camera], images[camera].shape[:2])
            self.identities[camera] = (
                np.arange(len(sampled[camera][0])), sampled[camera][1]
            )
            self.point_counts[camera] = len(sampled[camera][0])
            self.buffers[camera] = deque([images[camera].copy()])

        max_points = max(self.point_counts.values())
        padded_height = max(shape[0] for shape in self.shapes.values())
        padded_width = max(shape[1] for shape in self.shapes.values())
        queries = np.zeros((len(self.camera_names), max_points, 3), dtype=np.float32)
        for batch_index, camera in enumerate(self.camera_names):
            points = sampled[camera][0]
            queries[batch_index, :len(points), 1:] = points
        self.predictor = self.predictor_factory().to(self.device).eval()
        self._fix_batched_model_stride()
        queries = self._append_support_grid(queries, padded_height, padded_width)
        query_tensor = torch.from_numpy(queries).to(self.device)
        self.predictor(self._batch_video(), is_first_step=True, queries=query_tensor,
                       add_support_grid=False)
        if self.predictor.step != self.step or self.predictor.model.window_len != self.window_size or getattr(self.predictor, "v2", False):
            raise RuntimeError("Expected CoTracker3 online with window_size=16 and step=8")
        self.predictors = {camera: self.predictor for camera in self.camera_names}
        self.frame_index = 0
        return None

    @torch.inference_mode()
    def initialize_points(self, images: Mapping[str, np.ndarray], selections):
        """Initialize the batched predictor from per-camera named pixel queries."""
        if self.frame_index >= 0:
            raise RuntimeError("Already initialized; call reset() for a new stream")
        selected = direct_point_queries(images, selections, self.camera_names)
        for camera in self.camera_names:
            points, names = selected[camera]
            self.shapes[camera] = images[camera].shape
            self.identities[camera] = (np.arange(len(points)), names)
            self.point_counts[camera] = len(points)
            self.buffers[camera] = deque([images[camera].copy()])

        max_points = max(self.point_counts.values())
        height = max(shape[0] for shape in self.shapes.values())
        width = max(shape[1] for shape in self.shapes.values())
        queries = np.zeros((len(self.camera_names), max_points, 3), dtype=np.float32)
        for batch_index, camera in enumerate(self.camera_names):
            points = selected[camera][0]
            queries[batch_index, :len(points), 1:] = points
        self.predictor = self.predictor_factory().to(self.device).eval()
        self._fix_batched_model_stride()
        queries = self._append_support_grid(queries, height, width)
        self.predictor(
            self._batch_video(), is_first_step=True,
            queries=torch.from_numpy(queries).to(self.device), add_support_grid=False,
        )
        if (self.predictor.step != self.step
                or self.predictor.model.window_len != self.window_size
                or getattr(self.predictor, "v2", False)):
            raise RuntimeError("Expected CoTracker3 online with window_size=16 and step=8")
        self.predictors = {camera: self.predictor for camera in self.camera_names}
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
            self.buffers[camera].append(frame.copy())
        self.updated = False
        self.last_inference_ms = {}
        self.frame_index += 1
        if len(self.buffers[self.camera_names[0]]) < self.window_size:
            return self.get_latest_keypoints()
        if self.inference_count == 0:
            print("Starting batched CoTracker3 online inference: 16 frames accumulated.", flush=True)
        try:
            started = time.perf_counter()
            tracks, visibility = self.predictor(
                self._batch_video(), is_first_step=False, add_support_grid=False
            )
            elapsed = (time.perf_counter() - started) * 1000
            self.last_inference_ms = {"batch": elapsed}
            if (tracks is None or tracks.ndim != 4 or tracks.shape[0] != len(self.camera_names)
                    or tracks.shape[1] != self.frame_index + 1 or tracks.shape[2] < max(self.point_counts.values())):
                raise RuntimeError("CoTracker returned an unexpected timeline")
            tracks = tracks.detach().cpu().numpy()
            visibility = visibility.detach().cpu().numpy().astype(bool)
            if visibility.shape[:3] != tracks.shape[:3]:
                raise RuntimeError("CoTracker returned mismatched visibility")
            results = {}
            for batch_index, camera in enumerate(self.camera_names):
                count = self.point_counts[camera]
                points = tracks[batch_index, :, :count].copy()
                visible = visibility[batch_index, :, :count].copy()
                height, width = self.shapes[camera][:2]
                visible &= np.isfinite(points).all(axis=-1) & (points >= 0).all(axis=-1) & (points < [width, height]).all(axis=-1)
                point_ids, names = self.identities[camera]
                self.tracks[camera] = points
                self.visibility[camera] = visible
                results[camera] = TrackFrame(self.frame_index, points[-1].copy(), visible[-1].copy(), point_ids.copy(), names.copy(), "inferred")
            for camera in self.camera_names:
                for _ in range(self.step):
                    self.buffers[camera].popleft()
        except Exception:
            self.closed = True
            raise
        self.inference_count += 1
        self.inference_times_ms.append(dict(self.last_inference_ms))
        self.updated = True
        self.latest = results
        return self.get_latest_keypoints()


class OnlineTrackingPipeline(_OnlineTrackingPipeline):
    """Pipeline-compatible alias using the batched segmenter and tracker."""

    pass


__all__ = ["SAM3Segmenter", "OnlineKeypointTracker", "OnlineTrackingPipeline", "TrackFrame"]
