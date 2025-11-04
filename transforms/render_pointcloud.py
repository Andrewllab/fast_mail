from __future__ import annotations

import multiprocessing as mp
import traceback

import numpy as np
import open3d as o3d
import open3d.visualization as o3dvis
import torch
from tensordict import NonTensorData, TensorDict
from torch_geometric.data import Data

from environments.specs import CameraSpec, DataSpecs, PointCloudSpec
from transforms.base_transform import Transform
from utils.math import quaternion_to_matrix

ctx = mp.get_context("spawn")


class RenderPointCloud(ctx.Process, Transform):
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

        super().__init__(daemon=True)

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

        self.width = width
        self.height = height
        self.frame_size = pose_frame_size

        self.render_camera_poses = render_camera_poses
        self.render_ee_pose = render_ee_pose
        self.render_action = render_action

        self._specs = specs

        self.child_pipe, self.parent_pipe = ctx.Pipe(duplex=False)
        self.start()

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        nt_data: NonTensorData = tensordict["obs"].get(self._input_key)

        data: Data = nt_data.data  # unpack NonTensorData wrapper around pyg Data object
        # points, color = data_to_o3d(data)

        points = data.pos
        assert points is not None
        points = points.cpu().numpy()  # rendering of cuda tensors is not supported

        if (color := data.x) is not None:
            color = color.cpu().numpy()

        self.send_to_child(
            (
                "update_geometry",
                {"type": "PointCloud", "name": "pcd", "points": points, "color": color},
            )
        )

        if self.render_camera_poses:
            for key, spec in self._specs.obs.items():
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
                    # sometimes the dynamic extrinsics have been cleaned up, so just skip
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

                self.send_to_child(
                    (
                        "update_geometry",
                        {
                            "type": "CoordinateFrame",
                            "name": key,
                            "translation": translation,
                            "rotation": rotation,
                        },
                    )
                )

        if self.render_ee_pose:
            ee_pose = tensordict["obs", "ee_pose"].cpu()
            # remove the batch dimension and index the last element
            translation = ee_pose[0, -1, :3].numpy()
            # `quaternion_to_matrix` requires a batch dimension, so leave it in
            rotation = quaternion_to_matrix(ee_pose[0, -1:, 3:])
            rotation = rotation.squeeze(dim=0).numpy()

            self.send_to_child(
                (
                    "update_geometry",
                    {
                        "type": "CoordinateFrame",
                        "name": "ee_pose",
                        "translation": translation,
                        "rotation": rotation,
                    },
                )
            )

        if self.render_action:
            action = tensordict["action"].cpu()
            # remove the batch dimension and index the last element
            translation = action[0, -1, :3].numpy()
            # `quaternion_to_matrix` requires a batch dimension, so leave it in
            rotation = quaternion_to_matrix(action[0, -1:, 3:7])
            rotation = rotation.squeeze(dim=0).numpy()

            self.send_to_child(
                (
                    "update_geometry",
                    {
                        "type": "CoordinateFrame",
                        "name": "action",
                        "translation": translation,
                        "rotation": rotation,
                    },
                )
            )

        return tensordict

    def send_to_child(self, msg: object) -> None:
        try:
            self.parent_pipe.send(msg)
        except (BrokenPipeError, EOFError):
            # rendering process has crashed or exited
            raise KeyboardInterrupt("Open3D visualizer quit")

    def run(self) -> None:

        vis = o3dvis.Visualizer()
        vis.create_window(f"obs.{self._input_key}", self.width, self.height)
        geometries = {}

        # add an extra large coordinate frame at the origin
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.frame_size * 3, origin=[0, 0, 0]
        )
        vis.add_geometry(origin)
        geometries["origin"] = origin

        if self.render_camera_poses:
            # add coordinate frames for cameras
            for key, spec in self.specs.obs.items():
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

                vis.add_geometry(camera_frame)
                geometries[key] = camera_frame

        try:
            while True:
                # Non-blocking check for incoming messages
                if self.child_pipe.poll():
                    cmd, data = self.child_pipe.recv()
                    if cmd == "QUIT":
                        break

                    elif cmd == "update_geometry":
                        match data["type"]:
                            case "PointCloud":
                                name = data["name"]
                                points = data["points"]
                                points = o3d.utility.Vector3dVector(points)
                                color = data["color"]
                                if color is not None:
                                    if color.dtype == np.uint8:
                                        color = color.astype(np.float32) / 255.0
                                    color = o3d.utility.Vector3dVector(color)

                                if name in geometries:
                                    pcd = geometries[name]
                                    pcd.points = points
                                    if color is not None:
                                        pcd.colors = color
                                    vis.update_geometry(pcd)

                                else:
                                    # on first call
                                    pcd = o3d.geometry.PointCloud(points)
                                    if color is not None:
                                        pcd.colors = color
                                    vis.add_geometry(pcd)
                                    geometries[name] = pcd

                            case "CoordinateFrame":
                                name = data["name"]
                                translation = data["translation"]
                                rotation = data["rotation"]

                                # since we can't set an absolute pose, remove the old coordinate
                                # frame and add a new one with the correct pose
                                try:
                                    frame = geometries[name]
                                    vis.remove_geometry(frame, reset_bounding_box=False)
                                except KeyError:
                                    pass

                                frame = (
                                    o3d.geometry.TriangleMesh.create_coordinate_frame(
                                        size=self.frame_size,
                                        origin=translation,
                                    )
                                )
                                frame.rotate(rotation, center=translation)
                                vis.add_geometry(frame, reset_bounding_box=False)
                                geometries[name] = frame

                            case _:
                                print(f"Unknown geometry type: {data['type']}")

                    else:
                        print(f"Unknown command: {cmd}")

                # Keep the window responsive
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

            self.child_pipe.close()

    def close(self) -> None:
        self.send_to_child(("QUIT", {}))
        self.terminate()
        self.join()
        self.parent_pipe.close()


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
