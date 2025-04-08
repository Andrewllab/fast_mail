import logging
import time

import open3d as o3d
import open3d.core as o3c
import open3d.visualization as o3dvis
import open3d.visualization.rendering as rendering
from open3d.visualization import gui
from tensordict import NonTensorData
from torch_geometric.data import Data

from environments.specs import DataSpecs, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform
from utils.o3d import torch_to_o3d

log = logging.getLogger(__name__)


class RenderPointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        width: int = 1024,
        height: int = 768,
        fps: float | None = None,
    ) -> None:
        input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, PointCloudSpec)
        }

        if len(input_specs) == 0:
            raise ValueError("No point clouds found in specs")
        elif len(input_specs) > 1:
            raise ValueError(
                "RenderPointCloud only supports one point cloud key at a time"
            )

        self._input_key, self._input_spec = next(iter(input_specs.items()))

        # must initialize app singleton before creating any windows
        self.app = gui.Application.instance
        self.app.initialize()

        vis = o3dvis.O3DVisualizer(f"obs.{self._input_key}", width, height)
        vis.show_axes = True
        vis.show_settings = True
        vis.show_skybox(False)
        vis.reset_camera_to_default()
        self.vis = vis
        self.app.add_window(vis)

        mesh_frame = o3d.t.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=[0, 0, 0]
        )
        self.vis.add_geometry("origin", mesh_frame)
        self.static_geometries = [mesh_frame]

        # store the point cloud here and update it in-place
        self.pcd = None

        # render flags are required to tell the visualizer what to update
        self._render_flags = rendering.Scene.UPDATE_POINTS_FLAG
        if self._input_spec.color:
            self._render_flags |= rendering.Scene.UPDATE_COLORS_FLAG

        self.fps = fps

        self._output_specs = specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [KeyMapping(in_keys=[("obs", self._input_key)], out_keys=["_"])]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(self, data: NonTensorData) -> None:
        data = data.data  # unpack NonTensorData wrapper around pyg Data object
        points, color = data_to_o3dtensor(data)

        if self.pcd is None:
            pcd = o3d.t.geometry.PointCloud(points)
            if color is not None:
                pcd.point.colors = color

            self.vis.add_geometry("pcd", pcd)
            self.pcd = pcd

        else:
            self.pcd.point.positions = points
            if color:
                self.pcd.point.colors = color

            self.vis.update_geometry("pcd", self.pcd, self._render_flags)

        running = self.app.run_one_tick()
        if not running:
            log.info("Open3D app quit")
            raise KeyboardInterrupt

        if self.fps is not None:
            time.sleep(1 / self.fps)

    def close(self) -> None:
        self.app.quit()


def data_to_o3dtensor(data: Data) -> tuple[o3c.Tensor, o3c.Tensor | None]:
    """
    Convert a Data object to an Open3D PointCloud.
    """
    data = data.cpu()  # rendering of cuda tensors is not supported

    points = data.pos
    assert points is not None
    # remove the batch dimension and index the last element in the sequence
    points = points[0, -1]
    points = torch_to_o3d(points)

    if data.x is not None:
        # TODO: ensure color is in [0, 1]
        color = data.x
        color = color[0, -1]
        color = torch_to_o3d(color)
    else:
        color = None

    return points, color
