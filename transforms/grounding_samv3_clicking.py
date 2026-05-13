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
    "target object": {"color": "tab:red",  "key": "1"},
    "tool object":   {"color": "tab:cyan", "key": "2"},
}
DEFAULT_CLASS_ORDER = list(CLASS_CONFIG.keys())


# =============================================================================
# Stage 1 — Click collection
# =============================================================================

@dataclass
class _ClickAnnotations:
    """
    points[camera_key][class_name] = list of (x, y, label) tuples.
    label = 1 (positive / foreground) or 0 (negative / background).
    """
    points: Dict[str, Dict[str, List[Tuple[int, int, int]]]] = field(default_factory=dict)

    def ensure(self, camera_keys: List[str], class_names: List[str]):
        for cam in camera_keys:
            self.points.setdefault(cam, {})
            for cls in class_names:
                self.points[cam].setdefault(cls, [])

    def is_empty(self) -> bool:
        return not any(pts for by_cls in self.points.values() for pts in by_cls.values())

    def get(self, camera_key: str, class_name: str) -> List[Tuple[int, int, int]]:
        return self.points.setdefault(camera_key, {}).setdefault(class_name, [])

    def to_yaml_dict(self) -> Dict[str, Any]:
        return {
            "cameras": {
                cam: {
                    cls: [{"x": int(x), "y": int(y), "label": int(lbl)}
                          for (x, y, lbl) in pts]
                    for cls, pts in by_cls.items()
                }
                for cam, by_cls in self.points.items()
            }
        }

    @classmethod
    def from_yaml_dict(cls, d: Dict[str, Any]) -> "_ClickAnnotations":
        out = cls()
        for cam, by_cls in (d.get("cameras") or {}).items():
            out.points[cam] = {}
            for cls_name, pts in (by_cls or {}).items():
                out.points[cam][cls_name] = [
                    (int(p["x"]), int(p["y"]), int(p.get("label", 1)))
                    for p in (pts or [])
                ]
        return out


class _ClickCollectorGUI:
    """
    Modal matplotlib window for collecting clicks on one frame per camera.

    Controls
    --------
    left click   : add positive point of current class to clicked image
    shift+left   : add negative point of current class
    right click  : remove nearest point on clicked image
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
        existing: Optional[_ClickAnnotations] = None,
        max_remove_distance: float = 20.0,
        title_suffix: str = "",
    ):
        if not camera_keys:
            raise ValueError("camera_keys must be non-empty")
        for cam in camera_keys:
            if cam not in camera_frames:
                raise ValueError(f"missing frame for camera '{cam}'")
        for c in class_names:
            if c not in CLASS_CONFIG:
                raise ValueError(
                    f"Unknown class '{c}'. Known: {list(CLASS_CONFIG)}"
                )

        self.camera_frames = {k: self._to_uint8(v) for k, v in camera_frames.items()}
        self.camera_keys = list(camera_keys)
        self.class_names = list(class_names)
        self.max_remove_distance = max_remove_distance
        self.title_suffix = title_suffix

        self.annotations = existing or _ClickAnnotations()
        self.annotations.ensure(self.camera_keys, self.class_names)

        self.current_class = self.class_names[0]
        self.last_touched_camera = self.camera_keys[0]
        self.confirmed = False
        self._warned_missing = False

        # matplotlib handles, set on launch
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

        # Radio buttons for class selection
        ax_radio = self.fig.add_axes([0.87, 0.55, 0.11, 0.20])
        self.radio = RadioButtons(ax_radio, self.class_names, active=0)
        self.radio.on_clicked(self._on_radio_change)
        ax_radio.set_title("Class")

        # Action buttons
        Button(self.fig.add_axes([0.10, 0.07, 0.10, 0.06]), "Undo"
               ).on_clicked(lambda e: self._undo())
        Button(self.fig.add_axes([0.21, 0.07, 0.10, 0.06]), "Clear"
               ).on_clicked(lambda e: self._clear())
        Button(self.fig.add_axes([0.78, 0.07, 0.18, 0.06]), "Confirm & Close"
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
        is_shift = (getattr(event, "key", None) or "").lower().startswith("shift")

        if event.button == 1:
            label = 0 if is_shift else 1
            self.annotations.get(camera, self.current_class).append((x, y, label))
            sign = "negative" if label == 0 else "positive"
            self._set_status(
                f"Added {sign} {self.current_class} click on {camera} at ({x}, {y})"
            )
            self._redraw()
        elif event.button == 3:
            self._remove_nearest(camera, x, y)

    def _on_key(self, event):
        if event.key is None:
            return
        k = event.key.lower()
        # numeric class switching: '1' .. 'N'
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
                x, y, _ = pts.pop()
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
            for idx, (px, py, _) in enumerate(self.annotations.get(camera, cls)):
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
                pts = self.annotations.get(cam, cls)
                if not any(lbl == 1 for (_, _, lbl) in pts):
                    missing.append((cam, cls))
        return missing

    def _confirm(self):
        missing = self._missing_classes()
        if missing and not self._warned_missing:
            msg = "Missing positive clicks for: " + ", ".join(
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
                ax.legend(by_label.values(), by_label.keys(),
                          loc="upper right", fontsize=8)
        self._refresh_title()
        self.fig.canvas.draw_idle()

    def _plot_points(self, ax, camera: str):
        for cls in self.class_names:
            pts = self.annotations.get(camera, cls)
            if not pts:
                continue
            color = CLASS_CONFIG[cls]["color"]
            pos = [(x, y) for (x, y, lbl) in pts if lbl == 1]
            neg = [(x, y) for (x, y, lbl) in pts if lbl == 0]
            if pos:
                xs, ys = zip(*pos)
                ax.scatter(xs, ys, s=80, c=color, edgecolors="white",
                           linewidths=1.4, marker="o", label=f"{cls} (+)")
            if neg:
                xs, ys = zip(*neg)
                ax.scatter(xs, ys, s=90, c=color, edgecolors="white",
                           linewidths=1.4, marker="X", label=f"{cls} (-)")


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
    cache_yaml          : optional YAML file to persist clicks across runs.
    reuse_cached_clicks : if True and cache_yaml exists, skip the GUI on init.
                          A reset=True call always re-prompts regardless.
    """

    def __init__(
        self,
        specs: DataSpecs,
        class_names: Optional[str | List[str]] = None,
        camera_keys: str | List[str] = "left_cam",
        rgb_key: str = "rgb",
        cache_yaml: Optional[str | Path] = None,
        reuse_cached_clicks: bool = True,
        max_remove_distance_px: float = 20.0,
    ):
        super().__init__()
        self._specs = specs

        self.camera_keys = (
            list(camera_keys)
            if isinstance(camera_keys, (list, ListConfig))
            else [camera_keys]
        )
        if class_names is None:
            self.class_names = list(DEFAULT_CLASS_ORDER)
        else:
            self.class_names = (
                list(class_names)
                if isinstance(class_names, (list, ListConfig))
                else [class_names]
            )

        self.rgb_key = rgb_key
        self.cache_yaml = Path(cache_yaml) if cache_yaml is not None else None
        self.reuse_cached_clicks = reuse_cached_clicks
        self.max_remove_distance_px = max_remove_distance_px

        # Cached clicks (kept in memory so we can re-write them every call
        # without reopening the GUI).
        self._cached: Optional[_ClickAnnotations] = None

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

    def _load_cached_clicks(self) -> Optional[_ClickAnnotations]:
        if self.cache_yaml is None or not self.cache_yaml.exists():
            return None
        try:
            with self.cache_yaml.open("r", encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
            return _ClickAnnotations.from_yaml_dict(d)
        except Exception as e:
            print(f"[ClickCollector] failed to load cache {self.cache_yaml}: {e}")
            return None

    def _save_clicks(self, ann: _ClickAnnotations):
        if self.cache_yaml is None:
            return
        self.cache_yaml.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_yaml.open("w", encoding="utf-8") as f:
            yaml.safe_dump(ann.to_yaml_dict(), f, sort_keys=False)

    def _collect_clicks_via_gui(self, tensordict: TensorDict) -> _ClickAnnotations:
        frames = {
            cam: self._extract_first_frame(tensordict, cam)
            for cam in self.camera_keys
        }
        gui = _ClickCollectorGUI(
            camera_frames=frames,
            camera_keys=self.camera_keys,
            class_names=self.class_names,
            existing=None,
            max_remove_distance=self.max_remove_distance_px,
            title_suffix="left=add  shift+left=negative  right=remove  enter=confirm",
        )
        return gui.collect()

    def _write_clicks_to_td(
        self, tensordict: TensorDict, ann: _ClickAnnotations
    ):
        """
        Flatten the per-class click structure into per-point tensors keyed by
        ('clicks', camera_key, ...). Empty cameras get empty tensors.
        """
        for cam in self.camera_keys:
            points: List[Tuple[int, int]] = []
            labels: List[int] = []
            obj_ids: List[int] = []
            for cls_idx, cls in enumerate(self.class_names):
                obj_id = cls_idx + 1
                for (x, y, lbl) in ann.get(cam, cls):
                    points.append((x, y))
                    labels.append(lbl)
                    obj_ids.append(obj_id)

            if points:
                pts_t = torch.tensor(points, dtype=torch.long)
                lab_t = torch.tensor(labels, dtype=torch.long)
                obj_t = torch.tensor(obj_ids, dtype=torch.long)
            else:
                pts_t = torch.zeros((0, 2), dtype=torch.long)
                lab_t = torch.zeros((0,), dtype=torch.long)
                obj_t = torch.zeros((0,), dtype=torch.long)

            tensordict["clicks", cam, "points"] = pts_t
            tensordict["clicks", cam, "labels"] = lab_t
            tensordict["clicks", cam, "obj_ids"] = obj_t

    # ---- main entry ----

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if tensordict.shape[0] != 1:
            raise ValueError(
                "ClickPromptCollectorTransform expects batch_size=1"
            )
        reset = bool(tensordict.get("reset", False))
        need_collect = reset or self._cached is None

        if need_collect:
            ann: Optional[_ClickAnnotations] = None
            if not reset and self.reuse_cached_clicks:
                ann = self._load_cached_clicks()
                if ann is not None and not ann.is_empty():
                    print(
                        f"[ClickCollector] reusing cached clicks from {self.cache_yaml}"
                    )
            if ann is None or ann.is_empty():
                ann = self._collect_clicks_via_gui(tensordict)
                if ann.is_empty():
                    raise RuntimeError(
                        "No clicks were collected; refusing to proceed."
                    )
                self._save_clicks(ann)
            self._cached = ann

        self._write_clicks_to_td(tensordict, self._cached)
        return tensordict


# =============================================================================
# Stage 2 — SAM3 segmenter that consumes clicks from the tensordict
# =============================================================================

class ClickPromptSamV3VideoSegmenterTransform(Transform):
    """
    Persistent SAM3 video tracker driven by point clicks read from the
    tensordict. Modeled directly on PersistentSamV3VideoSegmenterTransform,
    but with two changes:

      1. No GroundingDINO — prompts come from
         tensordict["clicks", camera_key, {"points","labels","obj_ids"}]
         which a preceding ClickPromptCollectorTransform is expected to
         populate.
      2. Multiple objects per camera are registered explicitly via obj_ids;
         each unique obj_id maps to one entry in segmenter_out_keys
         (obj_id k+1 → segmenter_out_keys[k]).

    Lifecycle is identical to PersistentSamV3VideoSegmenterTransform:
        first call (or reset=True) → init session per camera
        subsequent calls           → stream frame through existing session
    """

    def __init__(
        self,
        specs: DataSpecs,
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
        self.out_device = (torch.device(out_device)
                           if isinstance(out_device, str) else out_device)

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
        self.segmenter_out_keys = (
            list(segmenter_out_keys)
            if isinstance(segmenter_out_keys, (list, ListConfig))
            else [segmenter_out_keys]
        )

        self.rgb_key = rgb_key
        self.verbose = verbose

        # Per-camera SAM video sessions. None means "needs init on next call".
        self.video_sessions: Dict[str, Any] = {cam: None for cam in self.camera_keys}
        # Per-camera: list of obj_ids actually registered (in registration
        # order). Needed to map model outputs back to segmenter_out_keys.
        self._registered_obj_ids: Dict[str, List[int]] = {
            cam: [] for cam in self.camera_keys
        }

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    # ---- helpers ----

    def _extract_frame(self, tensordict: TensorDict, camera_key: str):
        """Return [H,W,C] frame as whatever type the SAM processor accepts."""
        frame = tensordict["obs"][camera_key][self.rgb_key]
        if hasattr(frame, "shape") and len(frame.shape) == 4:
            frame = frame[0]
        return frame

    def _read_clicks(
        self, tensordict: TensorDict, camera_key: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            pts = tensordict["clicks", camera_key, "points"]
            lab = tensordict["clicks", camera_key, "labels"]
            obj = tensordict["clicks", camera_key, "obj_ids"]
        except KeyError as e:
            raise KeyError(
                f"No clicks found at ('clicks', '{camera_key}', ...). "
                "Did you run ClickPromptCollectorTransform first?"
            ) from e
        # Tensors are stored at batch_size=1; squeeze the leading batch dim if any.
        if pts.ndim == 3:
            pts = pts[0]
            lab = lab[0]
            obj = obj[0]
        return pts, lab, obj

    def _init_session_for_camera(
        self,
        camera_key: str,
        first_frame,
        points: torch.Tensor,   # [N, 2] long
        labels: torch.Tensor,   # [N]    long
        obj_ids: torch.Tensor,  # [N]    long
    ):
        """
        Build a fresh streaming SAM3 video session and register one object per
        unique obj_id using its point clicks.
        """
        inputs = self.sam_processor(
            images=first_frame, device=self.device, return_tensors="pt"
        )
        video_session = self.sam_processor.init_video_session(
            inference_device=self.device,
            dtype=self.sam_model.dtype,
        )

        registered: List[int] = []
        # Stable order: ascending obj_id, so output ordering is deterministic.
        unique_ids = sorted({int(x.item()) for x in obj_ids})
        for oid in unique_ids:
            mask = (obj_ids == oid)
            obj_points = points[mask].tolist()    # [[x, y], ...]
            obj_labels = labels[mask].tolist()    # [1, 0, 1, ...]
            if not obj_points:
                continue
            self.sam_processor.add_inputs_to_inference_session(
                inference_session=video_session,
                frame_idx=0,
                obj_ids=oid,
                input_points=[[obj_points]],    # [image][object][points]
                input_labels=[[obj_labels]],    # [image][object][labels]
                original_size=inputs.original_sizes[0],
            )
            registered.append(oid)

        if not registered:
            raise RuntimeError(
                f"No clicks were registered for camera '{camera_key}'."
            )

        self.video_sessions[camera_key] = video_session
        self._registered_obj_ids[camera_key] = registered

    # ---- main entry ----

    @torch.no_grad()
    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if tensordict.shape[0] != 1:
            raise ValueError(
                "ClickPromptSamV3VideoSegmenterTransform expects batch_size=1"
            )

        reset = bool(tensordict.get("reset", False))
        for cam in self.camera_keys:
            session = self.video_sessions[cam]
            need_init = reset or session is None

            frame = self._extract_frame(tensordict, cam)

            if need_init:
                pts, lab, obj = self._read_clicks(tensordict, cam)
                self._init_session_for_camera(cam, frame, pts, lab, obj)
                session = self.video_sessions[cam]

            inputs = self.sam_processor(
                images=frame, device=self.device, return_tensors="pt"
            )
            out = self.sam_model(
                inference_session=session,
                frame=inputs.pixel_values[0],
            )
            masks_per_obj = self.sam_processor.post_process_masks(
                [out.pred_masks],
                original_sizes=inputs.original_sizes,
                binarize=True,
            )[0]  # [num_obj, H, W] or [num_obj, 1, H, W]
            if masks_per_obj.ndim == 4:
                masks_per_obj = masks_per_obj[:, 0]

            # Map model output position → segmenter_out_keys
            # masks_per_obj[i] corresponds to self._registered_obj_ids[cam][i].
            # obj_id `k+1` → segmenter_out_keys[k]. Missing obj_ids → zero mask.
            H, W = self._frame_hw(frame)
            registered = self._registered_obj_ids[cam]
            id_to_mask = {oid: masks_per_obj[i] for i, oid in enumerate(registered)}

            for k, out_key in enumerate(self.segmenter_out_keys):
                desired_oid = k + 1
                if desired_oid in id_to_mask:
                    m = id_to_mask[desired_oid] > 0
                else:
                    m = torch.zeros((H, W), dtype=torch.bool, device=self.device)
                tensordict["obs", cam, out_key] = m.to(self.out_device)

                if self.verbose:
                    print(
                        f"[ClickSegmenter] {cam}/{out_key} (obj_id={desired_oid}): "
                        f"{int(m.sum())} mask pixels"
                    )

        return tensordict

    @staticmethod
    def _frame_hw(frame) -> Tuple[int, int]:
        if isinstance(frame, torch.Tensor):
            shape = tuple(frame.shape)
        else:
            shape = np.asarray(frame).shape
        # [H, W, C]
        return int(shape[0]), int(shape[1])