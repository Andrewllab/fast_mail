from __future__ import annotations

import dataclasses
import itertools
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

from environments.specs import DepthCameraSpec, PointCloudSpec, RGBCameraSpec
from transforms.base_transform import Transform
from utils.math import transform_pointmap, unproject_depth

if TYPE_CHECKING:
    from tensordict import TensorDict

    from environments.specs import DataSpecs


"""In ROS, the camera is looking down the +Z axis with the +Y axis pointing down,
and +X axis pointing right. This is the convention the point clouds are in after
conversion by `unproject_depth`. On the other hand, the typical world coordinate
system is with +X pointing forward, +Y pointing left, and +Z pointing up. We can
achieve this transformation using the following rotation matrix.

Reference: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#isaaclab.utils.math.convert_camera_frame_orientation_convention

(equivalent to T_USD_to_WORLD @ (T_USD_to_ROS)^(-1) in the convention used in Isaac Sim)
"""
ROS_TO_WORLD = [
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
]


class ToPointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        color: bool = False,
        max_depth: float | None = None,
        out_key: str = "pcd",
    ):
        self.color = color
        self.max_depth = max_depth
        self._out_key = out_key

        depth_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, DepthCameraSpec)
        }
        for key, spec in depth_specs.items():
            if spec.intrinsics is None:
                raise ValueError(
                    f"Depth camera spec {key} does not have an intrinsics matrix."
                )

            if color and not isinstance(spec, RGBCameraSpec):
                raise ValueError(
                    f"Depth camera spec {key} is not an RGBCameraSpec. Cannot use color."
                )

        # correct extrinsics by adding conversion from ROS to WORLD camera convention
        has_extrinsics = [spec.extrinsics is not None for spec in depth_specs.values()]
        if all(has_extrinsics):
            for key, spec in depth_specs.items():
                extrinsics = spec.extrinsics
                assert extrinsics is not None
                # we right-multiply, since we first need to transform the points
                # into the WORLD convention, and then apply the extrinsics
                extrinsics[:3, :3] = extrinsics[:3, :3] @ torch.tensor(
                    ROS_TO_WORLD, dtype=extrinsics.dtype
                )

                depth_specs[key] = dataclasses.replace(spec, extrinsics=extrinsics)
        elif any(has_extrinsics):
            raise ValueError(
                "All depth cameras must have extrinsics or none of them must have extrinsics."
            )
        elif any(
            spec.dynamic_pose_obs_key is not None for spec in depth_specs.values()
        ):
            raise ValueError(
                "Dynamic pose obs key is not supported for depth cameras without extrinsics."
            )

        self._input_specs = depth_specs

        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        obs_specs["pcd"] = PointCloudSpec(shape=(6 if color else 3,))
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        default_float_dtype = torch.get_default_dtype()

        # collect points, masks, and rgb from all cameras
        all_points = []
        all_masks = [] if self.max_depth is not None else None
        all_rgb = [] if self.color else None

        for key, spec in self._input_specs.items():

            subkey = spec.depth_subkeys[0]
            nested_key = ("obs", key)
            if subkey is not None:
                nested_key += (subkey,)
            depth = tensordict[nested_key]

            assert spec.intrinsics is not None
            # points: (..., H, W, 3)
            points = unproject_depth(
                depth,
                spec.intrinsics.intrinsic_matrix.to(depth.device),
                is_ortho=spec.orthogonal,
            )
            if (extrinsics := spec.extrinsics) is not None:
                extrinsics = extrinsics.to(points.device)

                if (pose_key := spec.dynamic_pose_obs_key) is not None:
                    if not isinstance(pose_key, tuple):
                        pose_key = (pose_key,)
                    dynamic_extrinsics = tensordict[("obs",) + pose_key]
                    assert dynamic_extrinsics.shape[-2:] == (4, 4)

                    # left multiply the dynamic extrinsics because they are
                    # applied after the static extrinsics
                    # (..., 4, 4) @ (4, 4) = (..., 4, 4)
                    extrinsics = dynamic_extrinsics @ extrinsics

                points = transform_pointmap(points, extrinsics)

            all_points.append(points)

            # remove points that are beyond the max depth
            if self.max_depth is not None:
                # get mask of points that are within max depth
                # mask: (..., H, W)
                mask = depth < self.max_depth
                all_masks.append(mask)

            if self.color:
                assert isinstance(spec, RGBCameraSpec)
                subkey = spec.rgb_subkeys[0]
                nested_key = ("obs", key)
                if subkey is not None:
                    nested_key += (subkey,)
                rgb = tensordict[nested_key]

                if spec.channel_order == "CHW":
                    # convert to HWC order
                    rgb = torch.movedim(rgb, -3, -1)

                # convert to float in range [0, 1]
                rgb = rgb.to(dtype=default_float_dtype).div(255)

                # rgb and depth must have the same resolution
                # rgb: (..., H, W, 3)
                assert rgb.shape[-3:-1] == depth.shape[-2:]

                all_rgb.append(rgb)

        # points: (..., H, W, 3) -> (..., N, H, W, 3)
        points_batch = torch.stack(all_points, dim=-4)
        # here we must assert that the time dimension is a singleton, since
        # we don't know how to handle time sequences of point clouds
        assert points_batch.ndim == 6
        if points_batch.shape[1] != 1:
            raise ValueError(
                f"Point cloud has time dimension {points_batch.shape[1]} (shape: {points_batch.shape}). Time sequences of point clouds are not supported."
            )
        points_batch = points_batch.squeeze(1)

        if self.max_depth is not None:
            # mask: (..., H, W) -> (..., N, H, W)
            mask_batch = torch.stack(all_masks, dim=-3)
            mask_batch = mask_batch.squeeze(1)
        else:
            mask_batch = None

        if self.color:
            # rgb: (..., H, W, 3) -> (..., N, H, W, 3)
            rgb_batch = torch.stack(all_rgb, dim=-4)
            rgb_batch = rgb_batch.squeeze(1)
        else:
            rgb_batch = None

        batch = collate_points(points_batch, mask_batch, rgb_batch)

        tensordict["obs", self._out_key] = batch
        return tensordict


def collate_points(
    points_batch: Tensor,
    mask_batch: Tensor | None = None,
    rgb_batch: Tensor | None = None,
) -> Batch:
    # TODO: refactor this to create a Batch object directly instead of using
    # Batch.from_data_list, which presumably makes a copy

    mask_batch_ = mask_batch if mask_batch is not None else itertools.repeat(None)
    rgb_batch_ = rgb_batch if rgb_batch is not None else itertools.repeat(None)

    datas = []
    for points, mask, rgb in zip(points_batch, mask_batch_, rgb_batch_):
        if mask is not None:
            # remove points that are beyond the max depth, flattening in the process
            points = points[mask]
            if rgb is not None:
                rgb = rgb[mask]

        else:
            # flatten point maps from all cameras into one point cloud
            points = points.flatten(end_dim=-2)
            if rgb is not None:
                rgb = rgb.flatten(end_dim=-2)

        # x refers to "node features" in torch geometric
        data = Data(pos=points, x=rgb)

        datas.append(data)

    return Batch.from_data_list(datas)
