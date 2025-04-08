import numpy as np
import open3d as o3d
import open3d.visualization as o3dvis
from tensordict import NonTensorData

from environments.specs import DataSpecs, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform


class RenderPointCloud(Transform):
    def __init__(self, specs: DataSpecs) -> None:
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
        vis.create_window()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.zeros(((100,) + self._input_spec.shape), dtype=np.float32)
        )
        vis.add_geometry(pcd)
        self.pcd = pcd

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.2, origin=[0, 0, 0]
        )
        vis.add_geometry(mesh_frame)
        self.static_geometries = [mesh_frame]

        self.vis = vis

        self._output_specs = specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [KeyMapping(in_keys=[("obs", self._input_key)], out_keys=["_"])]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(self, data: NonTensorData) -> None:
        data = data.data  # unpack NonTensorData wrapper around pyg Data object
        points = data.pos

        self.pcd.points = o3d.utility.Vector3dVector(
            points[..., :3].reshape(-1, 3).cpu().numpy()
        )
        if self._input_spec.color:
            self.pcd.colors = o3d.utility.Vector3dVector(
                points[..., 3:6].reshape(-1, 3).cpu().numpy()
            )

        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self) -> None:
        self.vis.destroy_window()
