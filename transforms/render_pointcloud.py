import time

import open3d as o3d
import open3d.visualization as o3dvis
from tensordict import TensorDict
from torch_geometric.data import Data

from environments.specs import CameraSpec, DataSpecs, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform


class RenderPointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        width: int = 1024,
        height: int = 768,
        fps: float | None = None,
        render_coordinate_frames: bool = False,
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

        vis = o3dvis.Visualizer()
        vis.create_window(f"obs.{self._input_key}", width, height)
        self.vis = vis

        self.geometries = {}

        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        self.vis.add_geometry(origin)
        self.geometries["origin"] = origin

        if render_coordinate_frames:
            # add coordinate frames for cameras
            for key, spec in specs.obs.items():
                if not isinstance(spec, CameraSpec) or spec.extrinsics is None:
                    continue

                key = f"{key}_origin"
                extrinsics = spec.extrinsics
                rotation = extrinsics[:3, :3]
                translation = extrinsics[:3, 3]

                camera = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1,
                    origin=translation,
                )
                camera.rotate(rotation, center=translation)

                self.vis.add_geometry(camera)
                self.geometries[key] = camera
        self._render_coordinate_frames = render_coordinate_frames

        self.fps = fps

        self._output_specs = specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [KeyMapping(in_keys=[("obs", self._input_key)], out_keys=["_"])]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        data = tensordict["obs"].get(self._input_key)

        data = data.data  # unpack NonTensorData wrapper around pyg Data object
        points, color = data_to_o3d(data)

        if "pcd" not in self.geometries:
            pcd = o3d.geometry.PointCloud(points)
            if color is not None:
                pcd.point.colors = color

            self.vis.add_geometry(pcd)
            self.geometries["pcd"] = pcd

        else:
            pcd = self.geometries["pcd"]
            pcd.points = points
            if color is not None:
                pcd.colors = color

            self.vis.update_geometry(pcd)

        self.vis.poll_events()
        self.vis.update_renderer()

        if self.fps is not None:
            time.sleep(1 / self.fps)

        return tensordict

    def close(self) -> None:
        self.vis.destroy_window()


def data_to_o3d(
    data: Data,
) -> tuple[o3d.utility.Vector3dVector, o3d.utility.Vector3dVector | None]:
    """
    Convert a Data object to an Open3D PointCloud.
    """
    data = data.cpu()  # rendering of cuda tensors is not supported

    points = data.pos
    assert points is not None
    # remove the batch dimension and index the last element in the sequence
    points = points[0, -1]
    points = o3d.utility.Vector3dVector(points.numpy())

    if data.x is not None:
        # TODO: ensure color is in [0, 1]
        color = data.x
        color = color[0, -1]
        color = o3d.utility.Vector3dVector(color.numpy())
    else:
        color = None

    return points, color
