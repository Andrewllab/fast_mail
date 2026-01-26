from __future__ import annotations

import logging
import multiprocessing as mp
import traceback
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import open3d as o3d
import open3d.visualization as o3dvis
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import Transform

log = logging.getLogger(__name__)
ctx = mp.get_context("spawn")


class RenderPointCloud(ctx.Process, Transform):
    """
    Multiprocess Open3D pointcloud renderer.

    - Uses the *selection + pointcloud creation* approach from `pakt_render_pointcloud.py`
      (multiple keys, key paths split on '/', uniform per-cloud colors, points from td[key][0]).
    - Uses the *rendering setup* approach from `render_pointcloud.py`
      (dedicated spawned process, Visualizer window kept responsive, updates via Pipe).

    This design is more robust in larger ML pipelines because Open3D's window/event loop
    lives in its own process.
    """

    def __init__(
        self,
        specs: DataSpecs,
        *,
        width: int = 1024,
        height: int = 768,
        point_size: float = 5.0,
        line_width: float = 2.0,
        show_ui: bool = True,
        log_pointcloud_size: bool = False,
        pcd_keys: str | Sequence[str] = "pcd",
        window_name: str | None = None,
    ) -> None:
        super().__init__(daemon=True)

        self._specs = specs

        # Selection logic (matches pakt_render_pointcloud.py)
        self._input_keys = pcd_keys
        if isinstance(self._input_keys, str):
            self._input_keys = [self._input_keys]
        self._input_keys = [k.split("/") for k in self._input_keys]

        self.width = int(width)
        self.height = int(height)

        self.point_size = float(point_size)
        self.line_width = float(line_width)
        self.show_ui = bool(show_ui)

        self.log_pointcloud_size = bool(log_pointcloud_size)

        # Same palette as pakt_render_pointcloud.py
        self.colors: List[List[float]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
        ]

        self._window_name = window_name or "pakt_render_pointcloud_multiprocess"

        # One-way pipe: parent -> child
        self.child_pipe, self.parent_pipe = ctx.Pipe(duplex=False)
        self.start()

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        """
        Collects pointcloud tensors from the provided keys and streams them to the
        render process.
        """
        for i, key_parts in enumerate(self._input_keys):
            name = "/".join(key_parts)

            try:
                pts = tensordict[tuple(key_parts)]
            except KeyError:
                log.warning(f"Key '{name}' not found in tensordict!")
                continue

            # Pointcloud creation semantics (matches pakt_render_pointcloud.py):
            # - take the first element in batch dimension
            # - convert to CPU numpy for Open3D
            pts_np = pts[0].detach().cpu().numpy()

            if pts_np.ndim != 2 or pts_np.shape[-1] != 3:
                log.warning(
                    f"Key '{name}' has unexpected shape {tuple(pts_np.shape)}; expected [N,3] (after batch index). Skipping."
                )
                continue

            if self.log_pointcloud_size:
                log.info(f"Pointcloud '{name}' has {len(pts_np)} points.")

            # Uniform per-cloud color (matches pakt_render_pointcloud.py),
            # but we send per-point colors since that's what Open3D PointCloud expects.
            rgb = np.asarray(self.colors[i % len(self.colors)], dtype=np.float64)
            colors_np = np.tile(rgb[None, :], (pts_np.shape[0], 1))

            self.send_to_child(
                (
                    "update_geometry",
                    {
                        "type": "PointCloud",
                        "name": name,
                        "points": pts_np,
                        "colors": colors_np,
                    },
                )
            )

        return tensordict

    def send_to_child(self, msg: object) -> None:
        try:
            self.parent_pipe.send(msg)
        except (BrokenPipeError, EOFError):
            raise KeyboardInterrupt("Open3D visualizer quit")

    def run(self) -> None:
        vis = o3dvis.Visualizer()
        vis.create_window(self._window_name, self.width, self.height)

        geometries: Dict[str, o3d.geometry.Geometry] = {}

        # add an extra large coordinate frame at the origin
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0])
        vis.add_geometry(origin)
        geometries["origin"] = origin

        # Render options (analogous to pakt_render_pointcloud.py vis_options)
        try:
            ro = vis.get_render_option()
            if ro is not None:
                ro.point_size = float(self.point_size)
                ro.line_width = float(self.line_width)
        except Exception:
            # Render option access can fail on some platforms; don't crash the renderer.
            pass

        try:
            while True:
                if self.child_pipe.poll():
                    cmd, data = self.child_pipe.recv()
                    if cmd == "QUIT":
                        break

                    if cmd != "update_geometry":
                        log.warning(f"Unknown command: {cmd}")
                        continue

                    if data.get("type") != "PointCloud":
                        log.warning(f"Unknown geometry type: {data.get('type')}")
                        continue

                    name = str(data["name"])
                    points_np = np.asarray(data["points"], dtype=np.float64)
                    colors_np = np.asarray(data["colors"], dtype=np.float64)

                    points = o3d.utility.Vector3dVector(points_np)
                    colors = o3d.utility.Vector3dVector(colors_np)

                    if name in geometries:
                        pcd = geometries[name]
                        assert isinstance(pcd, o3d.geometry.PointCloud)
                        pcd.points = points
                        pcd.colors = colors
                        vis.update_geometry(pcd)
                    else:
                        pcd = o3d.geometry.PointCloud(points)
                        pcd.colors = colors
                        vis.add_geometry(pcd)
                        geometries[name] = pcd

                # Keep window responsive
                vis.poll_events()
                vis.update_renderer()

        except KeyboardInterrupt:
            pass
        except Exception:
            traceback.print_exc()
        finally:
            try:
                vis.destroy_window()
            except Exception:
                pass
            try:
                self.child_pipe.close()
            except Exception:
                pass

    def close(self) -> None:
        try:
            self.send_to_child(("QUIT", {}))
        except Exception:
            pass
        try:
            self.terminate()
        except Exception:
            pass
        try:
            self.join()
        except Exception:
            pass
        try:
            self.parent_pipe.close()
        except Exception:
            pass
