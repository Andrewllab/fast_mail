#!/usr/bin/env python3
"""
HDF5 point annotation tool for preparing SAM/SAMv3 prompts.

Features
--------
- Load multiple HDF5 files
- Visualize left/right RGB camera frames side by side
- Switch files and timesteps
- Mark multiple pixels for two classes:
    1 -> target object
    2 -> tool object
- Click directly into either image to add points
- Remove nearest point with right click or `d`
- Undo last point with `u`
- Save/load annotations to YAML
- Export flattened CSV
- Optional filtering of input files via glob

Default HDF5 datasets
---------------------
left cam:  hf["obs"]["left_cam"]["frames"]["left"]
right cam: hf["obs"]["right_cam"]["frames"]["left"]

Dependencies
------------
pip install h5py matplotlib pyyaml numpy pandas
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.widgets import Button, RadioButtons

DEFAULT_LEFT_PATH = "obs/left_cam/frames/left"
DEFAULT_RIGHT_PATH = "obs/right_cam/frames/left"

CLASS_CONFIG = {
    "target object": {"color": "tab:red", "key": "1"},
    "tool object": {"color": "tab:cyan", "key": "2"},
}
CLASS_ORDER = list(CLASS_CONFIG.keys())


def nested_get(hf: h5py.File, slash_path: str):
    cur = hf
    for part in slash_path.strip("/").split("/"):
        cur = cur[part]
    return cur


def discover_files(inputs: List[str]) -> List[Path]:
    out: List[Path] = []
    for item in inputs:
        matches = [Path(p) for p in glob.glob(item)]
        if matches:
            out.extend(matches)
        else:
            p = Path(item)
            if p.exists():
                out.append(p)
    unique_sorted = sorted(set(out))
    if not unique_sorted:
        raise FileNotFoundError("No HDF5 files found from the provided inputs.")
    return unique_sorted


@dataclass
class SessionFileInfo:
    path: Path
    n_timesteps: int
    left_shape: Tuple[int, ...]
    right_shape: Tuple[int, ...]


class AnnotationStore:
    """
    YAML schema:

    files:
      /abs/path/file1.h5:
        timesteps:
          "0":
            left_cam:
              target object:
                - {x: 100, y: 200}
              tool object: []
            right_cam:
              target object: []
              tool object: []
    """

    def __init__(self):
        self.data: Dict[str, Any] = {"files": {}}

    @classmethod
    def load(cls, path: Path) -> "AnnotationStore":
        store = cls()
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Invalid YAML in {path}")
            store.data = loaded if "files" in loaded else {"files": loaded}
            store.data.setdefault("files", {})
        return store

    def ensure_frame(self, file_path: str, timestep: int):
        files = self.data.setdefault("files", {})
        fentry = files.setdefault(file_path, {})
        timesteps = fentry.setdefault("timesteps", {})
        tentry = timesteps.setdefault(str(timestep), {})
        for cam in ("left_cam", "right_cam"):
            centry = tentry.setdefault(cam, {})
            for cls_name in CLASS_ORDER:
                centry.setdefault(cls_name, [])

    def get_points(
        self, file_path: str, timestep: int, camera: str, cls_name: str
    ) -> List[Dict[str, int]]:
        self.ensure_frame(file_path, timestep)
        return self.data["files"][file_path]["timesteps"][str(timestep)][camera][
            cls_name
        ]

    def add_point(
        self, file_path: str, timestep: int, camera: str, cls_name: str, x: int, y: int
    ):
        points = self.get_points(file_path, timestep, camera, cls_name)
        points.append({"x": int(x), "y": int(y)})

    def pop_last(
        self, file_path: str, timestep: int, camera: str, cls_name: Optional[str] = None
    ) -> Optional[Tuple[str, Dict[str, int]]]:
        self.ensure_frame(file_path, timestep)
        if cls_name is not None:
            pts = self.get_points(file_path, timestep, camera, cls_name)
            if pts:
                pt = pts.pop()
                return cls_name, pt
            return None

        # Pop from whichever class has the latest appended point conceptually.
        # Since YAML doesn't track insertion timestamps, use class order reversed.
        for name in reversed(CLASS_ORDER):
            pts = self.get_points(file_path, timestep, camera, name)
            if pts:
                pt = pts.pop()
                return name, pt
        return None

    def remove_nearest(
        self,
        file_path: str,
        timestep: int,
        camera: str,
        x: float,
        y: float,
        max_dist_px: float = 20.0,
        cls_name: Optional[str] = None,
    ) -> Optional[Tuple[str, Dict[str, int], float]]:
        self.ensure_frame(file_path, timestep)
        best = None  # (dist, class_name, idx, point)
        class_names = [cls_name] if cls_name else CLASS_ORDER
        for name in class_names:
            if name is None:
                continue
            pts = self.get_points(file_path, timestep, camera, name)
            for idx, pt in enumerate(pts):
                dist = math.hypot(pt["x"] - x, pt["y"] - y)
                if best is None or dist < best[0]:
                    best = (dist, name, idx, pt)
        if best is None or best[0] > max_dist_px:
            return None
        dist, name, idx, pt = best
        pts = self.get_points(file_path, timestep, camera, name)
        removed = pts.pop(idx)
        return name, removed, dist

    def save_yaml(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.data, f, sort_keys=False)

    def save_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        files = self.data.get("files", {})
        for file_path, fentry in files.items():
            for timestep, tentry in fentry.get("timesteps", {}).items():
                for camera, centry in tentry.items():
                    for cls_name, points in centry.items():
                        for i, pt in enumerate(points):
                            rows.append(
                                {
                                    "file_path": file_path,
                                    "timestep": int(timestep),
                                    "camera": camera,
                                    "class_name": cls_name,
                                    "point_index": i,
                                    "x": int(pt["x"]),
                                    "y": int(pt["y"]),
                                }
                            )
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file_path",
                    "timestep",
                    "camera",
                    "class_name",
                    "point_index",
                    "x",
                    "y",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)


class H5AnnotationApp:
    def __init__(
        self,
        files: List[Path],
        annotations_yaml: Path,
        export_csv: Path,
        left_dataset_path: str = DEFAULT_LEFT_PATH,
        right_dataset_path: str = DEFAULT_RIGHT_PATH,
        max_remove_distance: float = 20.0,
    ):
        self.files = files
        self.annotations_yaml = annotations_yaml
        self.export_csv = export_csv
        self.left_dataset_path = left_dataset_path
        self.right_dataset_path = right_dataset_path
        self.max_remove_distance = max_remove_distance

        self.store = AnnotationStore.load(annotations_yaml)
        self.file_infos: List[SessionFileInfo] = self._inspect_files()

        self.file_idx = 0
        self.timestep = 0
        self.current_class = CLASS_ORDER[0]
        self.last_clicked_camera = "left_cam"
        self.status_text = None

        self.fig = None
        self.ax_left = None
        self.ax_right = None
        self.im_left = None
        self.im_right = None
        self.radio = None
        self.btn_prev_t = None
        self.btn_next_t = None
        self.btn_prev_f = None
        self.btn_next_f = None
        self.btn_save = None
        self.btn_undo = None

    def _inspect_files(self) -> List[SessionFileInfo]:
        infos = []
        for p in self.files:
            with h5py.File(p, "r") as hf:
                left = nested_get(hf, self.left_dataset_path)
                right = nested_get(hf, self.right_dataset_path)
                if left.ndim != 4 or left.shape[-1] != 3:
                    raise ValueError(
                        f"{p}: left dataset has shape {left.shape}, expected [T,H,W,3]"
                    )
                if right.ndim != 4 or right.shape[-1] != 3:
                    raise ValueError(
                        f"{p}: right dataset has shape {right.shape}, expected [T,H,W,3]"
                    )
                if left.shape[0] != right.shape[0]:
                    raise ValueError(
                        f"{p}: left/right number of timesteps differ: {left.shape[0]} vs {right.shape[0]}"
                    )
                infos.append(
                    SessionFileInfo(
                        path=p.resolve(),
                        n_timesteps=int(left.shape[0]),
                        left_shape=tuple(left.shape),
                        right_shape=tuple(right.shape),
                    )
                )
        return infos

    @property
    def current_file(self) -> Path:
        return self.file_infos[self.file_idx].path

    @property
    def current_info(self) -> SessionFileInfo:
        return self.file_infos[self.file_idx]

    def _read_current_images(self) -> Tuple[np.ndarray, np.ndarray]:
        with h5py.File(self.current_file, "r") as hf:
            left = np.asarray(nested_get(hf, self.left_dataset_path)[self.timestep])
            right = np.asarray(nested_get(hf, self.right_dataset_path)[self.timestep])

        left = self._normalize_for_display(left)
        right = self._normalize_for_display(right)
        return left, right

    @staticmethod
    def _normalize_for_display(img: np.ndarray) -> np.ndarray:
        if img.dtype == np.uint8:
            return img
        img = img.astype(np.float32)
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def launch(self):
        self.fig = plt.figure(figsize=(15, 8))
        self.ax_left = self.fig.add_axes([0.05, 0.18, 0.38, 0.72])
        self.ax_right = self.fig.add_axes([0.48, 0.18, 0.38, 0.72])

        ax_radio = self.fig.add_axes([0.88, 0.62, 0.11, 0.18])
        self.radio = RadioButtons(ax_radio, CLASS_ORDER, active=0)
        self.radio.on_clicked(self._on_radio_change)
        ax_radio.set_title("Class")

        self.btn_prev_t = Button(self.fig.add_axes([0.10, 0.06, 0.10, 0.06]), "Prev t")
        self.btn_next_t = Button(self.fig.add_axes([0.21, 0.06, 0.10, 0.06]), "Next t")
        self.btn_prev_f = Button(
            self.fig.add_axes([0.36, 0.06, 0.10, 0.06]), "Prev file"
        )
        self.btn_next_f = Button(
            self.fig.add_axes([0.47, 0.06, 0.10, 0.06]), "Next file"
        )
        self.btn_save = Button(self.fig.add_axes([0.66, 0.06, 0.10, 0.06]), "Save")
        self.btn_undo = Button(self.fig.add_axes([0.77, 0.06, 0.10, 0.06]), "Undo")

        self.btn_prev_t.on_clicked(lambda event: self._change_timestep(-1))
        self.btn_next_t.on_clicked(lambda event: self._change_timestep(+1))
        self.btn_prev_f.on_clicked(lambda event: self._change_file(-1))
        self.btn_next_f.on_clicked(lambda event: self._change_file(+1))
        self.btn_save.on_clicked(lambda event: self._save())
        self.btn_undo.on_clicked(lambda event: self._undo())

        self.status_text = self.fig.text(0.05, 0.01, "", fontsize=10)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._redraw()
        plt.show()

    def _set_status(self, text: str):
        if self.status_text is not None:
            self.status_text.set_text(text)
            self.fig.canvas.draw_idle()

    def _on_radio_change(self, label: str):
        self.current_class = label
        self._set_status(f"Selected class: {label}")

    def _change_timestep(self, delta: int):
        nt = self.current_info.n_timesteps
        self.timestep = int(np.clip(self.timestep + delta, 0, nt - 1))
        self._redraw()

    def _change_file(self, delta: int):
        self.file_idx = int(np.clip(self.file_idx + delta, 0, len(self.file_infos) - 1))
        self.timestep = int(
            np.clip(self.timestep, 0, self.current_info.n_timesteps - 1)
        )
        self._redraw()

    def _save(self):
        self.store.save_yaml(self.annotations_yaml)
        self.store.save_csv(self.export_csv)
        self._set_status(
            f"Saved YAML -> {self.annotations_yaml} | CSV -> {self.export_csv}"
        )

    def _undo(self):
        camera = self.last_clicked_camera
        res = self.store.pop_last(
            str(self.current_file),
            self.timestep,
            camera,
            cls_name=self.current_class,
        )
        if res is None:
            res = self.store.pop_last(
                str(self.current_file), self.timestep, camera, cls_name=None
            )

        if res is None:
            self._set_status(
                f"No points to undo on {camera} at timestep {self.timestep}"
            )
            return
        cls_name, pt = res
        self._set_status(
            f"Removed last point on {camera} | {cls_name}: ({pt['x']}, {pt['y']})"
        )
        self._redraw()

    def _on_key(self, event):
        if event.key is None:
            return
        k = event.key.lower()
        if k == "right":
            self._change_timestep(+1)
        elif k == "left":
            self._change_timestep(-1)
        elif k == "up":
            self._change_file(-1)
        elif k == "down":
            self._change_file(+1)
        elif k == "1":
            self.current_class = CLASS_ORDER[0]
            self.radio.set_active(0)
            self._set_status(f"Selected class: {self.current_class}")
        elif k == "2":
            self.current_class = CLASS_ORDER[1]
            self.radio.set_active(1)
            self._set_status(f"Selected class: {self.current_class}")
        elif k == "s":
            self._save()
        elif k == "u":
            self._undo()
        elif k == "d":
            # delete nearest point around mouse pointer if available
            x = getattr(event, "xdata", None)
            y = getattr(event, "ydata", None)
            ax = getattr(event, "inaxes", None)
            if x is None or y is None or ax not in (self.ax_left, self.ax_right):
                self._set_status(
                    "Move mouse over an image and press 'd' to delete nearest point"
                )
                return
            camera = "left_cam" if ax == self.ax_left else "right_cam"
            self.last_clicked_camera = camera
            removed = self.store.remove_nearest(
                str(self.current_file),
                self.timestep,
                camera,
                x,
                y,
                max_dist_px=self.max_remove_distance,
                cls_name=None,
            )
            if removed is None:
                self._set_status(
                    f"No nearby point within {self.max_remove_distance:.0f}px"
                )
                return
            cls_name, pt, dist = removed
            self._set_status(
                f"Removed {cls_name} point from {camera}: ({pt['x']}, {pt['y']}) [dist={dist:.1f}px]"
            )
            self._redraw()
        elif k == "h":
            self._print_help()

    def _print_help(self):
        help_text = (
            "Shortcuts:\n"
            "  1 / 2     -> switch class\n"
            "  left/right-> previous/next timestep\n"
            "  up/down   -> previous/next file\n"
            "  u         -> undo last point on last-clicked camera\n"
            "  d         -> delete nearest point under mouse\n"
            "  s         -> save YAML + CSV\n"
            "  h         -> print help to terminal\n"
            "Mouse:\n"
            "  left click  -> add point to clicked image\n"
            "  right click -> delete nearest point from clicked image\n"
        )
        print(help_text)
        self._set_status("Printed help to terminal")

    def _on_click(self, event):
        if event.inaxes not in (self.ax_left, self.ax_right):
            return
        if event.xdata is None or event.ydata is None:
            return

        camera = "left_cam" if event.inaxes == self.ax_left else "right_cam"
        self.last_clicked_camera = camera

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        if event.button == 1:
            self.store.add_point(
                str(self.current_file), self.timestep, camera, self.current_class, x, y
            )
            self._set_status(
                f"Added {self.current_class} point on {camera}: ({x}, {y})"
            )
            self._redraw()
        elif event.button == 3:
            removed = self.store.remove_nearest(
                str(self.current_file),
                self.timestep,
                camera,
                x,
                y,
                max_dist_px=self.max_remove_distance,
                cls_name=None,
            )
            if removed is None:
                self._set_status(
                    f"No nearby point within {self.max_remove_distance:.0f}px"
                )
                return
            cls_name, pt, dist = removed
            self._set_status(
                f"Removed {cls_name} point from {camera}: ({pt['x']}, {pt['y']}) [dist={dist:.1f}px]"
            )
            self._redraw()

    def _plot_points(self, ax, camera: str):
        file_path = str(self.current_file)
        for cls_name in CLASS_ORDER:
            pts = self.store.get_points(file_path, self.timestep, camera, cls_name)
            if not pts:
                continue
            xs = [pt["x"] for pt in pts]
            ys = [pt["y"] for pt in pts]
            color = CLASS_CONFIG[cls_name]["color"]
            ax.scatter(
                xs,
                ys,
                s=70,
                c=color,
                edgecolors="white",
                linewidths=1.2,
                marker="o",
                label=cls_name,
            )
            for idx, (x, y) in enumerate(zip(xs, ys)):
                ax.text(
                    x + 3,
                    y + 3,
                    str(idx),
                    color="white",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.18",
                        facecolor=color,
                        alpha=0.8,
                        edgecolor="none",
                    ),
                )

    def _redraw(self):
        left_img, right_img = self._read_current_images()

        self.ax_left.clear()
        self.ax_right.clear()

        self.ax_left.imshow(left_img)
        self.ax_right.imshow(right_img)

        self.ax_left.set_title("left_cam")
        self.ax_right.set_title("right_cam")

        self._plot_points(self.ax_left, "left_cam")
        self._plot_points(self.ax_right, "right_cam")

        for ax in (self.ax_left, self.ax_right):
            ax.set_axis_off()
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(
                    by_label.values(), by_label.keys(), loc="upper right", fontsize=8
                )

        self.fig.suptitle(
            f"File {self.file_idx + 1}/{len(self.file_infos)}: {self.current_file.name} | "
            f"Timestep {self.timestep}/{self.current_info.n_timesteps - 1} | "
            f"Class: {self.current_class}",
            fontsize=12,
        )
        self.fig.canvas.draw_idle()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Annotate HDF5 camera frames with point prompts for SAM/SAMv3."
    )
    p.add_argument(
        "inputs",
        nargs="+",
        help="HDF5 files or glob patterns, e.g. data/*.h5",
    )
    p.add_argument(
        "--annotations-yaml",
        type=Path,
        default=Path("samv3_point_annotations.yaml"),
        help="Where to store the nested YAML annotations.",
    )
    p.add_argument(
        "--export-csv",
        type=Path,
        default=Path("samv3_point_annotations.csv"),
        help="Where to store a flattened CSV export.",
    )
    p.add_argument(
        "--left-dataset-path",
        type=str,
        default=DEFAULT_LEFT_PATH,
        help=f"Slash-separated HDF5 path for left camera frames. Default: {DEFAULT_LEFT_PATH}",
    )
    p.add_argument(
        "--right-dataset-path",
        type=str,
        default=DEFAULT_RIGHT_PATH,
        help=f"Slash-separated HDF5 path for right camera frames. Default: {DEFAULT_RIGHT_PATH}",
    )
    p.add_argument(
        "--max-remove-distance",
        type=float,
        default=20.0,
        help="Maximum distance in pixels for removing the nearest point.",
    )
    return p


def main():
    args = build_argparser().parse_args()
    files = discover_files(args.inputs)

    print(f"Loaded {len(files)} file(s).")
    print("Controls:")
    print("  1 / 2       switch class")
    print("  left/right  prev/next timestep")
    print("  up/down     prev/next file")
    print("  left click  add point")
    print("  right click remove nearest point")
    print("  u           undo last point")
    print("  s           save")
    print("  h           print help")

    app = H5AnnotationApp(
        files=files,
        annotations_yaml=args.annotations_yaml,
        export_csv=args.export_csv,
        left_dataset_path=args.left_dataset_path,
        right_dataset_path=args.right_dataset_path,
        max_remove_distance=args.max_remove_distance,
    )
    app.launch()


if __name__ == "__main__":
    main()
