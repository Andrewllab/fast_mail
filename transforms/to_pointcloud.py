from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Data

from environments.specs import DepthCameraSpec, PointCloudSpec, RGBCameraSpec
from transforms.base_transform import Transform
from utils.math import apply_homogeneous_transform, unproject_depth

if TYPE_CHECKING:
    from tensordict import TensorDict

    from environments.specs import DataSpecs


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
        self._input_specs = depth_specs

        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        obs_specs["pcd"] = PointCloudSpec(shape=(6 if color else 3,))
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:

        all_points = []
        if self.color:
            all_rgb = []

        for key, spec in self._input_specs.items():

            subkey = spec.depth_subkeys[0]
            nested_key = ("obs", key)
            if subkey is not None:
                nested_key += (subkey,)
            depth = tensordict[nested_key]

            assert spec.intrinsics is not None
            points = unproject_depth(
                depth,
                spec.intrinsics.intrinsic_matrix.to(depth.device),
                is_ortho=spec.orthogonal,
            )

            if (extrinsics := spec.extrinsics) is not None:
                points = apply_homogeneous_transform(
                    points, extrinsics.to(points.device)
                )

            if (pose_key := spec.dynamic_pose_obs_key) is not None:
                if not isinstance(pose_key, tuple):
                    pose_key = (pose_key,)
                dynamic_extrinsics = tensordict[("obs",) + pose_key]
                dynamic_extrinsics = dynamic_extrinsics[:, -1]  # index final timestep
                assert dynamic_extrinsics.shape[-2:] == (4, 4)
                points = apply_homogeneous_transform(points, dynamic_extrinsics)

            # remove points that are beyond the max depth
            if self.max_depth is not None:
                # get mask of points that are within max depth
                # mask: (leading_dims, H*W)
                mask = (depth < self.max_depth).flatten(start_dim=-2)
                points = points[mask]

                # restore leading dimensions after masking
                leading_dims = depth.shape[:-2]
                points = points.view(*leading_dims, -1, 3)

            else:
                mask = Ellipsis

            all_points.append(points)

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

                # rgb and depth must have the same resolution
                assert rgb.shape[-3::-1] == depth.shape[-2:]

                rgb = rgb.flatten(start_dim=-3, end_dim=-2)

                # remove points that are beyond the max depth
                rgb = rgb[mask]
                # restore leading dimensions after masking
                rgb = rgb.view(*leading_dims, -1, 3)

                all_rgb.append(rgb)

        # stack points from all cameras together
        # if we have batches of timesteps, they are kept separate
        all_points = torch.cat(all_points, dim=-2)

        data = {"pos": all_points}

        if self.color:
            all_rgb = torch.cat(all_rgb, dim=-2)
            # x refers to "node features" in PyG
            data["x"] = all_rgb

        tensordict["obs", self._out_key] = Data(**data)
        return tensordict
