"""
Two-stage click-prompted SAM3 segmentation.

Stage 1 — ClickPromptCollectorTransform
    Opens a matplotlib window on the first call (or after `reset=True`)
    and collects point clicks per camera, per class. Writes the clicks
    into the tensordict and caches them to YAML if requested. On
    subsequent calls it simply re-injects the cached clicks; the window
    stays closed.

Stage 2 — ClickPromptSamV3VideoSegmenterTransform
    Reads the clicks from the tensordict, initializes a persistent SAM3
    tracker video session per camera the first time it sees them (or
    after `reset=True`), and streams subsequent frames through the
    existing session. Almost identical in shape to
    PersistentSamV3VideoSegmenterTransform, just with point prompts
    sourced from the tensordict instead of GroundingDINO boxes.

The two transforms communicate via:
    tensordict["clicks", camera_key, "points"]  → LongTensor [N, 2] (x, y)
    tensordict["clicks", camera_key, "labels"]  → LongTensor [N]    (1=pos, 0=neg)
    tensordict["clicks", camera_key, "obj_ids"] → LongTensor [N]    (object id, 1-indexed)

Object id `k+1` corresponds to segmenter_out_keys[k] downstream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from omegaconf import ListConfig
from tensordict import TensorDict
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

from environments.specs import DataSpecs
from transforms.base_transform import Transform

# -----------------------------------------------------------------------------
# Class definitions (display config only — segmenter cares about obj_ids, not names)
# -----------------------------------------------------------------------------

CLASS_CONFIG = {
    "target object": {"color": "tab:red", "key": "1"},
    "tool object": {"color": "tab:cyan", "key": "2"},
}
DEFAULT_CLASS_ORDER = list(CLASS_CONFIG.keys())

DEFAULT_CLASS_COLORS = {
    "target object": "tab:red",
    "tool object": "tab:cyan",
}


# =============================================================================
# Stage 1 — Click collection
# =============================================================================


@dataclass
class _ClickAnnotations:
    """
    points[camera_key][class_name] = list of (x, y) tuples (all positive).
    """

    points: Dict[str, Dict[str, List[Tuple[int, int]]]] = field(default_factory=dict)

    def ensure(self, camera_keys: List[str], class_names: List[str]):
        for cam in camera_keys:
            self.points.setdefault(cam, {})
            for cls in class_names:
                self.points[cam].setdefault(cls, [])

    def is_empty(self) -> bool:
        return not any(
            pts for by_cls in self.points.values() for pts in by_cls.values()
        )

    def get(self, camera_key: str, class_name: str) -> List[Tuple[int, int]]:
        return self.points.setdefault(camera_key, {}).setdefault(class_name, [])

    def to_yaml_dict(self) -> Dict[str, Any]:
        return {
            "cameras": {
                cam: {
                    cls: [{"x": int(x), "y": int(y)} for (x, y) in pts]
                    for cls, pts in by_cls.items()
                }
                for cam, by_cls in self.points.items()
            }
        }


class _ClickCollectorGUI:
    """
    Modal matplotlib window for collecting positive point clicks on one
    frame per camera.

    Controls
    --------
    left click   : add a point of the current class to the clicked image
    right click  : remove nearest point on the clicked image
    1 / 2 / ...  : switch class
    u            : undo last click on last-touched camera
    d            : delete nearest point under mouse
    enter / c    : confirm & close
    r            : clear all points
    h            : print help to terminal
    """

    def __init__(
        self,
        camera_frames: Dict[str, np.ndarray],
        camera_keys: List[str],
        class_names: List[str],
        class_colors: Dict[str, str],
        existing: Optional[_ClickAnnotations] = None,
        max_remove_distance: float = 20.0,
        title_suffix: str = "",
    ):
        if not camera_keys:
            raise ValueError("camera_keys must be non-empty")
        for cam in camera_keys:
            if cam not in camera_frames:
                raise ValueError(f"missing frame for camera '{cam}'")

        self.camera_frames = {k: self._to_uint8(v) for k, v in camera_frames.items()}
        self.camera_keys = list(camera_keys)
        self.class_names = list(class_names)
        self.class_colors = dict(class_colors)
        self.max_remove_distance = max_remove_distance
        self.title_suffix = title_suffix

        self.annotations = existing or _ClickAnnotations()
        self.annotations.ensure(self.camera_keys, self.class_names)

        self.current_class = self.class_names[0]
        self.last_touched_camera = self.camera_keys[0]
        self.confirmed = False
        self._warned_missing = False

        self.fig = None
        self.axes: Dict[str, plt.Axes] = {}
        self.status_text = None
        self.radio = None

    @staticmethod
    def _to_uint8(img) -> np.ndarray:
        arr = np.asarray(img)
        if arr.dtype == np.uint8:
            return arr
        arr = arr.astype(np.float32)
        if arr.max() <= 1.0:
            arr = arr * 255.0
        return np.clip(arr, 0, 255).astype(np.uint8)

    def collect(self) -> _ClickAnnotations:
        n_cams = len(self.camera_keys)
        fig_w = max(8, 6 * n_cams)
        self.fig = plt.figure(figsize=(fig_w, 7))

        ax_width = 0.80 / n_cams
        for i, cam in enumerate(self.camera_keys):
            left = 0.04 + i * (ax_width + 0.01)
            ax = self.fig.add_axes([left, 0.20, ax_width, 0.70])
            ax.set_title(cam)
            ax.set_axis_off()
            self.axes[cam] = ax

        ax_radio = self.fig.add_axes([0.87, 0.55, 0.11, 0.20])
        self.radio = RadioButtons(ax_radio, self.class_names, active=0)
        self.radio.on_clicked(self._on_radio_change)
        ax_radio.set_title("Class")

        Button(self.fig.add_axes([0.10, 0.07, 0.10, 0.06]), "Undo").on_clicked(
            lambda e: self._undo()
        )
        Button(self.fig.add_axes([0.21, 0.07, 0.10, 0.06]), "Clear").on_clicked(
            lambda e: self._clear()
        )
        Button(
            self.fig.add_axes([0.78, 0.07, 0.18, 0.06]), "Confirm & Close"
        ).on_clicked(lambda e: self._confirm())

        self.status_text = self.fig.text(0.04, 0.01, "", fontsize=10)

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        suptitle = "Click prompts for SAM3 — current class: {}"
        if self.title_suffix:
            suptitle += f"  |  {self.title_suffix}"
        self._suptitle_template = suptitle

        self._redraw()
        plt.show()  # blocks until window closes
        return self.annotations

    # ---- internal helpers ----

    def _set_status(self, text: str):
        if self.status_text is not None:
            self.status_text.set_text(text)
            self.fig.canvas.draw_idle()

    def _axis_to_camera(self, ax) -> Optional[str]:
        for cam, cam_ax in self.axes.items():
            if cam_ax is ax:
                return cam
        return None

    def _on_radio_change(self, label: str):
        self.current_class = label
        self._set_status(f"Class: {label}")
        self._refresh_title()

    def _on_click(self, event):
        camera = self._axis_to_camera(event.inaxes)
        if camera is None or event.xdata is None or event.ydata is None:
            return
        self.last_touched_camera = camera
        x, y = int(round(event.xdata)), int(round(event.ydata))

        if event.button == 1:
            self.annotations.get(camera, self.current_class).append((x, y))
            self._set_status(
                f"Added {self.current_class} click on {camera} at ({x}, {y})"
            )
            self._redraw()
        elif event.button == 3:
            self._remove_nearest(camera, x, y)

    def _on_key(self, event):
        if event.key is None:
            return
        k = event.key.lower()
        if k.isdigit():
            idx = int(k) - 1
            if 0 <= idx < len(self.class_names):
                self.current_class = self.class_names[idx]
                self.radio.set_active(idx)
                self._set_status(f"Class: {self.current_class}")
                return
        if k == "u":
            self._undo()
        elif k == "d":
            cam = self._axis_to_camera(event.inaxes)
            if cam is None or event.xdata is None or event.ydata is None:
                self._set_status("Hover over an image then press 'd'")
                return
            self._remove_nearest(cam, event.xdata, event.ydata)
        elif k in ("enter", "c"):
            self._confirm()
        elif k == "r":
            self._clear()
        elif k == "h":
            print(self.__class__.__doc__)
            self._set_status("Printed help to terminal")

    def _undo(self):
        cam = self.last_touched_camera
        order = [self.current_class] + [
            c for c in reversed(self.class_names) if c != self.current_class
        ]
        for cls in order:
            pts = self.annotations.get(cam, cls)
            if pts:
                x, y = pts.pop()
                self._set_status(f"Undid {cls} click on {cam} at ({x}, {y})")
                self._redraw()
                return
        self._set_status(f"Nothing to undo on {cam}")

    def _clear(self):
        for cam in self.camera_keys:
            for cls in self.class_names:
                self.annotations.get(cam, cls).clear()
        self._set_status("Cleared all clicks")
        self._redraw()

    def _remove_nearest(self, camera: str, x: float, y: float):
        best = None
        for cls in self.class_names:
            for idx, (px, py) in enumerate(self.annotations.get(camera, cls)):
                d = math.hypot(px - x, py - y)
                if best is None or d < best[0]:
                    best = (d, cls, idx)
        if best is None or best[0] > self.max_remove_distance:
            self._set_status(
                f"No point within {self.max_remove_distance:.0f}px on {camera}"
            )
            return
        d, cls, idx = best
        pt = self.annotations.get(camera, cls).pop(idx)
        self._set_status(
            f"Removed {cls} click on {camera}: ({pt[0]}, {pt[1]}) [d={d:.1f}px]"
        )
        self._redraw()

    def _missing_classes(self) -> List[Tuple[str, str]]:
        missing = []
        for cam in self.camera_keys:
            for cls in self.class_names:
                if not self.annotations.get(cam, cls):
                    missing.append((cam, cls))
        return missing

    def _confirm(self):
        missing = self._missing_classes()
        if missing and not self._warned_missing:
            msg = "Missing clicks for: " + ", ".join(
                f"{cam}/{cls}" for cam, cls in missing
            )
            self._set_status(msg + "  (press Confirm again to close anyway)")
            self._warned_missing = True
            return
        self.confirmed = True
        plt.close(self.fig)

    def _refresh_title(self):
        if self.fig is None:
            return
        self.fig.suptitle(
            self._suptitle_template.format(self.current_class), fontsize=12
        )

    def _redraw(self):
        for cam, ax in self.axes.items():
            ax.clear()
            ax.imshow(self.camera_frames[cam])
            ax.set_title(cam)
            ax.set_axis_off()
            self._plot_points(ax, cam)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(
                    by_label.values(), by_label.keys(), loc="upper right", fontsize=8
                )
        self._refresh_title()
        self.fig.canvas.draw_idle()

    def _plot_points(self, ax, camera: str):
        for cls in self.class_names:
            pts = self.annotations.get(camera, cls)
            if not pts:
                continue
            color = self.class_colors.get(cls, "tab:orange")
            xs, ys = zip(*pts)
            ax.scatter(
                xs,
                ys,
                s=80,
                c=color,
                edgecolors="white",
                linewidths=1.4,
                marker="o",
                label=cls,
            )


class ClickPromptCollectorTransform(Transform):
    """
    Collects user point clicks once per session (first call or after reset)
    and writes them into the tensordict for downstream transforms.

    Output tensordict entries (per camera_key):
        tensordict["clicks", camera_key, "points"]  → LongTensor [N, 2] (x, y)
        tensordict["clicks", camera_key, "labels"]  → LongTensor [N]    (1=pos, 0=neg)
        tensordict["clicks", camera_key, "obj_ids"] → LongTensor [N]    (1-indexed)

    Object id k+1 corresponds to class_names[k], which (by convention) should
    line up with the downstream segmenter's segmenter_out_keys[k].

    Parameters
    ----------
    class_names         : ordered list of class names; defines the obj_id mapping.
                          Default ["target object", "tool object"].
    camera_keys         : cameras to collect clicks on.
    rgb_key             : key under obs[camera] holding the RGB frame.
    """

    def __init__(
        self,
        specs: DataSpecs,
        class_names: Optional[str | List[str]],
        click_keys: Optional[str | List[str]],
        camera_keys: str | List[str] = "left_cam",
        rgb_key: str = "rgb",
        max_remove_distance_px: float = 20.0,
        class_colors: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self._specs = specs

        self.camera_keys = (
            list(camera_keys)
            if isinstance(camera_keys, (list, ListConfig))
            else [camera_keys]
        )
        self.class_names = (
            list(class_names)
            if isinstance(class_names, (list, ListConfig))
            else [class_names]
        )
        self.click_keys = (
            list(click_keys)
            if isinstance(click_keys, (list, ListConfig))
            else [click_keys]
        )
        if len(self.click_keys) != len(self.class_names):
            raise ValueError(
                "click_keys length must match class_names length "
                f"({len(self.click_keys)} vs {len(self.class_names)})"
            )
        self.rgb_key = rgb_key
        self.max_remove_distance_px = max_remove_distance_px

        # Cached clicks (kept in memory so we can re-write them every call
        # without reopening the GUI).
        self._cached: Optional[_ClickAnnotations] = None

        merged_colors = dict(DEFAULT_CLASS_COLORS)
        if class_colors:
            merged_colors.update(class_colors)
        self.class_colors = merged_colors

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    # ---- helpers ----

    def _extract_first_frame(
        self, tensordict: TensorDict, camera_key: str
    ) -> np.ndarray:
        frame = tensordict["obs"][camera_key][self.rgb_key]
        if hasattr(frame, "shape") and len(frame.shape) == 4:
            frame = frame[0]
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        return frame

    def _collect_clicks_via_gui(self, tensordict: TensorDict) -> _ClickAnnotations:
        frames = {
            cam: self._extract_first_frame(tensordict, cam) for cam in self.camera_keys
        }
        gui = _ClickCollectorGUI(
            camera_frames=frames,
            camera_keys=self.camera_keys,
            class_names=self.class_names,
            class_colors=self.class_colors,
            existing=None,
            max_remove_distance=self.max_remove_distance_px,
            title_suffix="left=add  shift+left=negative  right=remove  enter=confirm",
        )
        return gui.collect()

    def _write_clicks_to_td(self, tensordict: TensorDict, ann: _ClickAnnotations):
        """
        Write one tensor per (camera, click_key) under obs.
        Shape: [N, 2] LongTensor of (x, y), positives only. Empty class →
        zero-length tensor of shape [0, 2].
        """
        for cam in self.camera_keys:
            for cls, click_key in zip(self.class_names, self.click_keys):
                pts = ann.get(cam, cls)
                if pts:
                    pts_t = torch.tensor(pts, dtype=torch.long)
                else:
                    pts_t = torch.zeros((0, 2), dtype=torch.long)
                tensordict["obs", cam, click_key] = pts_t.unsqueeze(
                    0
                )  # add batch dim for convenience

    # ---- main entry ----

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        reset = bool(tensordict["obs"].get("reset", False))
        need_collect = reset or self._cached is None

        if need_collect:
            ann: Optional[_ClickAnnotations] = None
            if ann is None or ann.is_empty():
                ann = self._collect_clicks_via_gui(tensordict)
                if ann.is_empty():
                    raise RuntimeError("No clicks were collected; refusing to proceed.")
            self._cached = ann

        self._write_clicks_to_td(tensordict, self._cached)
        return tensordict


# =============================================================================
# Stage 2 — SAM3 segmenter that consumes clicks from the tensordict
# =============================================================================


class ClickPromptSamV3VideoSegmenterTransform(Transform):
    """
    Persistent SAM3 video tracker driven by point clicks read from the
    tensordict. Modeled on PersistentSamV3VideoSegmenterTransform, with
    GroundingDINO replaced by user-provided clicks.

    Tensordict input (per camera, per click_key):
        tensordict["obs", camera_key, click_key] → LongTensor [N, 2]  (x, y)

    `click_keys[k]` provides the clicks for `segmenter_out_keys[k]`. Each
    click_key is registered as one SAM object with obj_id k+1. If a
    click_key has zero clicks for a camera, that class is skipped on that
    camera and its output mask is all zeros.

    Lifecycle is identical to PersistentSamV3VideoSegmenterTransform:
        first call (or reset=True) → init session per camera
        subsequent calls           → stream frame through existing session
    """

    def __init__(
        self,
        specs: DataSpecs,
        click_keys: str | List[str],
        segmenter_out_keys: str | List[str],
        camera_keys: str | List[str] = "left_cam",
        rgb_key: str = "rgb",
        device: str | torch.device = "cuda",
        out_device: str | torch.device = "cpu",
        sam_dtype: torch.dtype = torch.bfloat16,
        sam_model_id: str = "facebook/sam3",
        verbose: bool = False,
    ):
        super().__init__()
        self._specs = specs
        self.device = torch.device(device) if isinstance(device, str) else device
        self.out_device = (
            torch.device(out_device) if isinstance(out_device, str) else out_device
        )

        self.sam_model = Sam3TrackerVideoModel.from_pretrained(sam_model_id).to(
            self.device, dtype=sam_dtype
        )
        self.sam_model.eval()
        self.sam_processor = Sam3TrackerVideoProcessor.from_pretrained(sam_model_id)
        self.sam_dtype = sam_dtype

        self.camera_keys = (
            list(camera_keys)
            if isinstance(camera_keys, (list, ListConfig))
            else [camera_keys]
        )
        self.click_keys = (
            list(click_keys)
            if isinstance(click_keys, (list, ListConfig))
            else [click_keys]
        )
        self.segmenter_out_keys = (
            list(segmenter_out_keys)
            if isinstance(segmenter_out_keys, (list, ListConfig))
            else [segmenter_out_keys]
        )
        if len(self.click_keys) != len(self.segmenter_out_keys):
            raise ValueError(
                "click_keys length must match segmenter_out_keys length "
                f"({len(self.click_keys)} vs {len(self.segmenter_out_keys)})"
            )

        # Per camera: indices into click_keys for classes that had clicks at
        # session init time. Used to map model output slots back to the right
        # segmenter_out_keys when some classes had no clicks.
        self._present_class_indices: Dict[str, List[int]] = {
            cam: [] for cam in self.camera_keys
        }

        self.rgb_key = rgb_key
        self.verbose = verbose

        # Per-camera SAM video sessions. None ⇒ needs init on next call.
        self.video_sessions: Dict[str, Any] = {cam: None for cam in self.camera_keys}

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    # ---- main entry: mirrors PersistentSamV3VideoSegmenterTransform.__call__ ----

    @torch.no_grad()
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if tensordict.shape[0] != 1:
            raise ValueError(
                "ClickPromptSamV3VideoSegmenterTransform only supports batch_size=1"
            )

        for camera_key in self.camera_keys:
            video_session = self.video_sessions[camera_key]
            reset = bool(tensordict["obs"].get("reset", False))

            video_frame = tensordict["obs"][camera_key][self.rgb_key][0]

            # SAM Processing — called ONCE per frame, used by both
            # add_inputs_to_inference_session (init path) and sam_model.
            inputs = self.sam_processor(
                images=video_frame, device=self.device, return_tensors="pt"
            )

            if video_session is None or reset:
                # Read clicks for each class from the tensordict. None entries
                # are kept as a placeholder so the position-to-class mapping
                # stays aligned with click_keys; they are filtered out before
                # the SAM call (just like None boxes in the reference impl).
                init_points: List[Optional[List[List[int]]]] = []
                for click_key in self.click_keys:
                    pts = tensordict["obs", camera_key, click_key]
                    if pts.ndim == 3:
                        pts = pts[0]
                    if pts.numel() == 0:
                        init_points.append(None)
                    else:
                        init_points.append(pts.to(torch.long).tolist())

                # Remember which click_key positions had clicks, so we can map
                # model outputs back to the correct segmenter_out_keys and
                # write zero masks for any class that had no clicks.
                self._present_class_indices[camera_key] = [
                    i for i, p in enumerate(init_points) if p is not None
                ]

                # Filter out empty classes. Mirrors the reference transform's
                # `[box.tolist() for box in init_boxes if box is not None]`.
                valid_points = [p for p in init_points if p is not None]
                input_points = [valid_points]  # [image][object][points][2]
                input_labels = [[[1] * len(p) for p in valid_points]]

                if valid_points:
                    # init samv3 session
                    video_session = self.sam_processor.init_video_session(
                        inference_device=self.device,
                        dtype=self.sam_model.dtype,
                    )
                    self.video_sessions[camera_key] = video_session

                    self.sam_processor.add_inputs_to_inference_session(
                        inference_session=video_session,
                        frame_idx=0,
                        obj_ids=list(range(1, len(valid_points) + 1)),
                        input_points=input_points,
                        input_labels=input_labels,
                        original_size=inputs.original_sizes[0],
                    )

            # If no clicks were ever provided for this camera, emit zero masks
            # for every output key and move on.
            if self.video_sessions[camera_key] is None:
                H, W = video_frame.shape[0], video_frame.shape[1]
                for segmenter_out_key in self.segmenter_out_keys:
                    tensordict["obs", camera_key, segmenter_out_key] = torch.zeros(
                        (H, W), dtype=torch.bool, device=self.out_device
                    )
                continue

            sam3_tracker_video_output = self.sam_model(
                inference_session=video_session, frame=inputs.pixel_values[0]
            )
            video_res_masks_ = self.sam_processor.post_process_masks(
                [sam3_tracker_video_output.pred_masks],
                original_sizes=inputs.original_sizes,
                binarize=True,
            )
            video_res_masks = video_res_masks_[0]

            # Map model outputs (indexed by position among *present* classes)
            # back to segmenter_out_keys. Absent classes get a zero mask.
            present = self._present_class_indices[camera_key]
            H, W = video_frame.shape[0], video_frame.shape[1]
            present_to_model_idx = {cls_idx: i for i, cls_idx in enumerate(present)}
            for k, segmenter_out_key in enumerate(self.segmenter_out_keys):
                if k in present_to_model_idx:
                    mask = video_res_masks[present_to_model_idx[k]]
                else:
                    mask = torch.zeros((H, W), dtype=torch.bool, device=self.device)
                tensordict["obs", camera_key, segmenter_out_key] = mask.to(
                    self.out_device
                )

                if self.verbose:
                    print(
                        f"[ClickSegmenter] {camera_key}/{segmenter_out_key}: "
                        f"{int(mask.sum())} mask pixels"
                    )
        return tensordict
