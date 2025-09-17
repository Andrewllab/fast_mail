from __future__ import annotations

import itertools
import logging

import torch
from tensordict import TensorDict
from torch import Tensor
from torch_geometric.data import Batch, Data

from environments.base_dataset import EmptyPointCloudError
from environments.specs import (
    CameraSpec,
    DataSpecs,
    DepthStream,
    PointCloudSpec,
    RGBStream,
)
from transforms.base_transform import Transform
from utils.math import transform_pointmap, unproject_depth

log = logging.getLogger(__name__)


class ToPointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        color: bool = False,
        max_depth: float | None = None,
        out_key: str = "pcd",
        warn_min_points: int = 100,
    ):
        self.color = color
        self.max_depth = max_depth
        self._out_key = out_key
        self.warn_min_points = warn_min_points
        self.error_min_points = False

        depth_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            and any(isinstance(stream, DepthStream) for stream in spec.streams.values())
        }

        multiview = len(depth_specs) > 1

        for key, spec in depth_specs.items():
            name, depth_stream = next(
                (name, stream)
                for name, stream in spec.streams.items()
                if isinstance(stream, DepthStream)
            )
            if depth_stream.intrinsics is None:
                raise ValueError(
                    f"Depth stream at {key}.{name} does not have an intrinsics matrix."
                )

            rgb_streams = [
                stream
                for stream in spec.streams.values()
                if isinstance(stream, RGBStream)
            ]

            if color and not rgb_streams:
                raise ValueError(
                    f"Depth camera spec {key} is not an RGBCameraSpec. Cannot use color."
                )
            elif color and len(rgb_streams) > 1:
                log.warning(
                    f"Depth camera spec {key} has multiple RGB streams. Using {rgb_streams[0]} to color the point cloud."
                )

            if multiview and spec.extrinsics is None:
                raise ValueError(
                    f"Depth camera {key} does not have an extrinsics matrix."
                )
            if spec.dynamic_pose_obs_key is not None and spec.extrinsics is None:
                raise ValueError(
                    f"Dynamic pose obs key {spec.dynamic_pose_obs_key} is not supported for depth cameras without extrinsics."
                )

        self._input_specs = depth_specs

        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        obs_specs["pcd"] = PointCloudSpec(feature_dim=(6 if color else 3), color=color)
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # collect points, masks, and rgb from all cameras
        all_points = []
        all_masks = []  # if self.max_depth is not None, populate with masks
        all_rgb = []  # if self.color is True, populate with rgb

        for key, spec in self._input_specs.items():

            name, depth_stream = next(
                (name, stream)
                for name, stream in spec.streams.items()
                if isinstance(stream, DepthStream)
            )
            depth = tensordict["obs", key, name]

            assert depth_stream.intrinsics is not None
            # points: (..., H, W, 3)
            points = unproject_depth(
                depth,
                depth_stream.intrinsics.intrinsic_matrix.to(depth.device),
                is_ortho=depth_stream.orthogonal,
            )
            if (extrinsics := spec.extrinsics) is not None:
                extrinsics = extrinsics.to(points.device)

                if (pose_key := spec.dynamic_pose_obs_key) is not None:
                    if not isinstance(pose_key, tuple):
                        pose_key = (pose_key,)
                    dynamic_extrinsics = tensordict[("obs",) + pose_key]
                    assert dynamic_extrinsics.shape[-2:] == (4, 4)
                    # BackCompat
                    dynamic_extrinsics = dynamic_extrinsics.to(dtype=torch.float32)

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
                name, rgb_stream = next(
                    (name, stream)
                    for name, stream in spec.streams.items()
                    if isinstance(stream, RGBStream)
                )
                rgb = tensordict["obs", key, name]

                if rgb_stream.channel_order == "CHW":
                    # convert to HWC order
                    rgb = torch.movedim(rgb, -3, -1)

                # rgb and depth must have the same resolution
                # rgb: (..., H, W, 3)
                assert rgb.shape[-3:-1] == depth.shape[-2:]

                all_rgb.append(rgb)

        # points: (..., H, W, 3) -> (..., N, H, W, 3)
        points_batch = torch.stack(all_points, dim=-4)

        # here we must enforce that the time dimension is a singleton, since
        # we don't know how to handle time sequences of point clouds
        if points_batch.ndim == 6 and points_batch.shape[1] != 1:
            raise ValueError(
                f"Point cloud has time dimension {points_batch.shape[1]} (shape: {points_batch.shape}). Time sequences of point clouds are not supported."
            )

        # flatten batch and time dimensions
        # points_batch: (..., N, H, W, 3) -> (B, N, H, W, 3)
        points_batch = torch.flatten(points_batch, end_dim=-5)

        if self.max_depth is not None:
            # mask: (..., H, W) -> (..., N, H, W)
            mask_batch = torch.stack(all_masks, dim=-3)
            mask_batch = torch.flatten(mask_batch, end_dim=-4)
        else:
            mask_batch = None

        if self.color:
            # rgb: (..., H, W, 3) -> (..., N, H, W, 3)
            rgb_batch = torch.stack(all_rgb, dim=-4)
            rgb_batch = torch.flatten(rgb_batch, end_dim=-5)
        else:
            rgb_batch = None

        batch = flatten_and_collate(points_batch, mask_batch, rgb_batch)

        num_points = batch.ptr[1:] - batch.ptr[:-1]
        if (num_points <= self.warn_min_points).any():
            log.warning(
                f"Some point clouds are empty after conversion to point cloud. This may be due to all points being beyond max_depth={self.max_depth}. The number of points per batch element is: {num_points}"
            )
            if self.error_min_points:
                raise EmptyPointCloudError("Aborting due to empty point clouds.")

        tensordict["obs", self._out_key] = batch
        return tensordict

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        error_min_points = self.error_min_points
        self.error_min_points = True
        tensordict = self(tensordict)
        self.error_min_points = error_min_points
        return tensordict


def flatten_and_collate(
    points_batch: Tensor,
    mask_batch: Tensor | None = None,
    rgb_batch: Tensor | None = None,
) -> Batch:
    masks = mask_batch if mask_batch is not None else itertools.repeat(None)
    rgbs = rgb_batch if rgb_batch is not None else itertools.repeat(None)

    datas = []
    for points, mask, rgb in zip(points_batch, masks, rgbs):
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
