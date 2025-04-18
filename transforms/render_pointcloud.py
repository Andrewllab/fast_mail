from __future__ import annotations

import open3d as o3d
import open3d.visualization as o3dvis
import torch
from tensordict import NonTensorData, TensorDict
from torch_geometric.data import Data

from environments.specs import CameraSpec, DataSpecs, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform


class RenderPointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        width: int = 1024,
        height: int = 768,
        render_coordinate_frames: bool = False,
        pcd_key: str = "pcd",
    ) -> None:

        self._input_key = pcd_key
        try:
            self._input_spec = specs.obs[pcd_key]
        except KeyError:
            raise ValueError(
                f"Key {pcd_key} not found in specs. Available keys: {list(specs.obs.keys())}"
            )
        if not isinstance(self._input_spec, PointCloudSpec):
            raise ValueError(
                f"Key {pcd_key} is not a point cloud spec. Found {self._input_spec.type}"
            )

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

        self._output_specs = specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [KeyMapping(in_keys=[("obs", self._input_key)], out_keys=["_"])]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        nt_data: NonTensorData = tensordict["obs"].get(self._input_key)

        data: Data = nt_data.data  # unpack NonTensorData wrapper around pyg Data object
        points, color = data_to_o3d(data)

        if "pcd" not in self.geometries:
            pcd = o3d.geometry.PointCloud(points)
            if color is not None:
                pcd.colors = color

            self.vis.add_geometry(pcd)
            self.geometries["pcd"] = pcd

        else:
            pcd = self.geometries["pcd"]
            pcd.points = points
            if color is not None:
                pcd.colors = color

            self.vis.update_geometry(pcd)

        if self._render_coordinate_frames:
            for key, spec in self._output_specs.obs.items():
                if (
                    not isinstance(spec, CameraSpec)
                    or spec.dynamic_pose_obs_key is None
                ):
                    continue

                key = f"{key}_origin"
                pose_key = spec.dynamic_pose_obs_key
                if not isinstance(pose_key, tuple):
                    pose_key = (pose_key,)
                dynamic_extrinsics = tensordict[("obs",) + pose_key]
                # remove the batch dimension and index the last element in the sequence
                dynamic_extrinsics = dynamic_extrinsics[0, -1].cpu()

                # chain the dynamic extrinsics with the static extrinsics
                assert spec.extrinsics is not None
                extrinsics = dynamic_extrinsics @ spec.extrinsics
                extrinsics = extrinsics.numpy()
                rotation = extrinsics[:3, :3]
                translation = extrinsics[:3, 3]

                # since we can't set an absolute pose, remove the old coordinate
                # frame and add a new one with the correct pose
                camera = self.geometries[key]
                self.vis.remove_geometry(camera, reset_bounding_box=False)

                camera = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.1,
                    origin=translation,
                )
                camera.rotate(rotation, center=translation)

                self.vis.add_geometry(camera, reset_bounding_box=False)
                self.geometries[key] = camera

        self.vis.poll_events()
        self.vis.update_renderer()

        return tensordict

    def close(self) -> None:
        self.vis.destroy_window()


def data_to_o3d(
    data: Data,
) -> tuple[o3d.utility.Vector3dVector, o3d.utility.Vector3dVector | None]:
    """
    Convert a Data object to an Open3D PointCloud.
    """
    points = data.pos.cpu()  # rendering of cuda tensors is not supported
    assert points is not None
    points = o3d.utility.Vector3dVector(points.numpy())

    if data.x is not None:
        color = data.x.cpu()
        if color.dtype == torch.uint8:
            color = color.float() / 255.0
        color = o3d.utility.Vector3dVector(color.numpy())
    else:
        color = None

    return points, color
