from __future__ import annotations

import logging
import multiprocessing as mp

import torch
from tensordict import NonTensorData, TensorDict
from torch_geometric.data import Batch, Data

from environments.specs import CameraSpec, DataSpecs, PointCloudSpec
from transforms.base_transform import Transform
from utils.math import quaternion_to_matrix
from utils.o3d import AsyncPcdRenderer

ctx = mp.get_context("spawn")
log = logging.getLogger(__name__)


class RenderPointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        width: int = 1024,
        height: int = 768,
        render_camera_poses: bool = False,
        render_ee_pose: bool = False,
        render_action: bool = False,
        coordinate_frame_size: float = 0.1,
        log_pointcloud_size: bool = False,
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

        self.renderer = AsyncPcdRenderer(
            width=width,
            height=height,
            window_name=f"obs.{self._input_key}",
            coordinate_frame_size=coordinate_frame_size,
            log_pointcloud_size=log_pointcloud_size,
        )

        keys_poses_to_render = []
        if render_ee_pose:
            keys_poses_to_render.append(("obs", "ee_pose"))
        if render_action:
            keys_poses_to_render.append(("action",))
        self.keys_poses_to_render = keys_poses_to_render

        if render_camera_poses:
            # Add coordinate frames for cameras with static extrinsics (no
            # dynamic pose obs key) immediately since they won't change.
            # Coordinate frames for moving cameras are rendered in each loop
            # iteration, so skip them here.

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

                extrinsics = spec.extrinsics
                rotation = extrinsics[:3, :3]
                translation = extrinsics[:3, 3]

                name = f"{key}_origin"
                self.renderer.render_coordinate_frame(
                    translation=translation, rotation=rotation, name=name
                )
        self.render_camera_poses = render_camera_poses

        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        nt_data: NonTensorData = tensordict["obs"].get(self._input_key)

        data: Data = nt_data.data  # unpack NonTensorData wrapper around pyg Data object

        if isinstance(data, Batch):
            data = data.get_example(0)  # just render the first point cloud in the batch

        pos, color = data.pos, data.x
        assert pos is not None
        pos = pos.cpu().numpy()  # rendering of cuda tensors is not supported

        if color is not None:
            color = color.cpu().numpy()

        self.renderer.render_pcd(pos, color, name=self._input_key)

        if self.render_camera_poses:
            # Add coordinate frames for moving cameras (with dynamic pose obs key)

            for key, spec in self._specs.obs.items():
                if (
                    not isinstance(spec, CameraSpec)
                    or spec.extrinsics is None
                    or spec.dynamic_pose_obs_key is None
                ):
                    continue

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
                dynamic_extrinsics = dynamic_extrinsics[0, -1]

                # chain the dynamic extrinsics with the static extrinsics
                assert spec.extrinsics is not None
                extrinsics = dynamic_extrinsics @ spec.extrinsics
                extrinsics = extrinsics.cpu().numpy()
                rotation = extrinsics[:3, :3]
                translation = extrinsics[:3, 3]

                name = f"{key}_origin"
                self.renderer.render_coordinate_frame(
                    translation=translation, rotation=rotation, name=name
                )

        for pose_key in self.keys_poses_to_render:
            pose = tensordict.get(pose_key)
            if pose is None:
                # if the data doesn't exist for any reason, just skip it
                continue

            # remove the batch dimension and index the last element in the sequence
            pose = pose[0, -1].cpu()

            translation = pose[:3].numpy()
            rotation = quaternion_to_matrix(pose[3:7].unsqueeze(0)).squeeze(0).numpy()

            name = ".".join(pose_key)
            self.renderer.render_coordinate_frame(
                translation=translation, rotation=rotation, name=name
            )

        return tensordict
