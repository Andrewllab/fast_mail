from __future__ import annotations

import open3d as o3d
import open3d.visualization as o3dvis
import torch
from tensordict import NonTensorData, TensorDict
from torch_geometric.data import Data

from environments.specs import CameraSpec, DataSpecs, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform
from utils.math import quaternion_to_matrix


class RenderPointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        width: int = 1024,
        height: int = 768,
        render_camera_poses: bool = False,
        render_ee_pose: bool = False,
        render_action: bool = False,
        pose_frame_size: float = 0.1,
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
        self.frame_size = pose_frame_size

        # add an extra large coordinate frame at the origin
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.frame_size * 3, origin=[0, 0, 0]
        )
        self.vis.add_geometry(origin)
        self.geometries["origin"] = origin

        if render_camera_poses:
            # add coordinate frames for cameras
            for key, spec in specs.obs.items():
                if (
                    not isinstance(spec, CameraSpec)
                    or spec.extrinsics is None
                    or spec.dynamic_pose_obs_key is not None
                ):
                    # we only need to render cameras that have extrinsics
                    # for moving cameras, we create a new coordinate frame in
                    # each loop iteration, so skip it here
                    continue

                key = f"{key}_origin"
                extrinsics = spec.extrinsics
                rotation = extrinsics[:3, :3]
                translation = extrinsics[:3, 3]

                camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=self.frame_size,
                    origin=translation,
                )
                camera_frame.rotate(rotation, center=translation)

                self.vis.add_geometry(camera_frame)
                self.geometries[key] = camera_frame

        self.render_camera_poses = render_camera_poses
        self.render_ee_pose = render_ee_pose
        self.render_action = render_action

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
            # on first call
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

        if self.render_camera_poses:
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
                dynamic_extrinsics = tensordict.get(("obs",) + pose_key)
                if dynamic_extrinsics is None:
                    continue

                # BackCompat
                dynamic_extrinsics = dynamic_extrinsics.to(dtype=torch.float32)
                # remove the batch dimension and index the last element in the sequence
                dynamic_extrinsics = dynamic_extrinsics[0, -1].cpu()

                # chain the dynamic extrinsics with the static extrinsics
                assert spec.extrinsics is not None
                extrinsics = dynamic_extrinsics @ spec.extrinsics
                extrinsics = extrinsics.cpu().numpy()
                rotation = extrinsics[:3, :3]
                translation = extrinsics[:3, 3]

                # since we can't set an absolute pose, remove the old coordinate
                # frame and add a new one with the correct pose
                try:
                    camera_frame = self.geometries[key]
                    self.vis.remove_geometry(camera_frame, reset_bounding_box=False)
                except KeyError:
                    pass

                camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=self.frame_size,
                    origin=translation,
                )
                camera_frame.rotate(rotation, center=translation)

                self.vis.add_geometry(camera_frame, reset_bounding_box=False)
                self.geometries[key] = camera_frame

        if self.render_ee_pose:
            ee_pose = tensordict["obs", "ee_pose"].cpu()
            # remove the batch dimension and index the last element
            translation = ee_pose[0, -1, :3].numpy()
            # `quaternion_to_matrix` requires a batch dimension, so leave it in
            rotation = quaternion_to_matrix(ee_pose[0, -1:, 3:])
            rotation = rotation.squeeze(dim=0).numpy()

            # since we can't set an absolute pose, remove the old coordinate
            # frame and add a new one with the correct pose
            try:
                ee_pose_frame = self.geometries["ee_pose"]
                self.vis.remove_geometry(ee_pose_frame, reset_bounding_box=False)
            except KeyError:
                pass

            ee_pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.frame_size,
                origin=translation,
            )
            ee_pose_frame.rotate(rotation, center=translation)

            self.vis.add_geometry(ee_pose_frame, reset_bounding_box=False)
            self.geometries["ee_pose"] = ee_pose_frame

        if self.render_action:
            action = tensordict["action"].cpu()
            # remove the batch dimension and index the last element
            translation = action[0, -1, :3].numpy()
            # `quaternion_to_matrix` requires a batch dimension, so leave it in
            rotation = quaternion_to_matrix(action[0, -1:, 3:7])
            rotation = rotation.squeeze(dim=0).numpy()

            # since we can't set an absolute pose, remove the old coordinate
            # frame and add a new one with the correct pose
            try:
                action_frame = self.geometries["action"]
                self.vis.remove_geometry(action_frame, reset_bounding_box=False)
            except KeyError:
                pass

            action_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=self.frame_size,
                origin=translation,
            )
            action_frame.rotate(rotation, center=translation)

            self.vis.add_geometry(action_frame, reset_bounding_box=False)
            self.geometries["action"] = action_frame

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
