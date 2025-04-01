from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Data

from environments.specs import CameraSpec, DepthCameraSpec, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform
from utils.math import apply_homogeneous_transform, unproject_depth

if TYPE_CHECKING:
    from torch import Tensor

    from environments.specs import DataSpecs


class ToPointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        color: bool = False,
        max_depth: float | None = None,
    ):
        self.color = color
        self.max_depth = max_depth

        self.cam_cfgs = {}
        in_keys = []

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

            cam_cfg = {
                "shape": spec.shape,
                "intrinsics": spec.intrinsics.intrinsic_matrix,
                "orthogonal": spec.orthogonal,
                "extrinsics": spec.extrinsics,
                "dynamic": spec.dynamic_pose_obs_key is not None,
            }

            # add the depth image to the list of inputs
            in_keys.append(("obs", key))

            if spec.dynamic_pose_obs_key is not None:
                # add the dynamic pose to the list of inputs
                pose_obs_key = spec.dynamic_pose_obs_key
                if isinstance(pose_obs_key, str):
                    pose_obs_key = (pose_obs_key,)
                in_keys.append(("obs",) + pose_obs_key)

            if self.color:
                rgb_key = spec.rgb_obs_key
                if rgb_key is None:
                    raise ValueError(
                        f"Depth camera spec {key} does not have a corresponding RGB spec."
                    )
                if isinstance(rgb_key, str):
                    rgb_key = (rgb_key,)

                # add the rgb image to the list of inputs
                in_keys.append(("obs",) + rgb_key)

                rgb_spec = specs.obs
                for key in rgb_key:
                    rgb_spec = rgb_spec[key]

                assert isinstance(rgb_spec, CameraSpec)

                rgb_shape = rgb_spec.shape
                if rgb_shape[-1] == 3:
                    cam_cfg["channel_order"] = "HWC"
                elif rgb_shape[-3] == 3:
                    cam_cfg["channel_order"] = "CHW"
                else:
                    raise ValueError(
                        f"RGB image in obs.{spec.rgb_obs_key} must have either HWC or CHW channel order. Got spec with shape {rgb_shape}"
                    )

            self.cam_cfgs[key] = cam_cfg

        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        obs_specs["pcd"] = PointCloudSpec(shape=(6 if color else 3,))
        self._specs = specs.replace(obs=obs_specs)

        self._key_mappings = [KeyMapping(in_keys=in_keys, out_keys=[("obs", "pcd")])]

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return self._key_mappings

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, *args: Tensor) -> Data:

        all_points = []
        if self.color:
            all_rgb = []

        for cam_cfg in self.cam_cfgs.values():
            # since we can't know in advance how many arguments we will have,
            # (and it varies by camera), we have to unpack each argument as we go
            depth, args = args[0], args[1:]
            assert depth.shape[-3:] == cam_cfg["shape"]

            points = unproject_depth(
                depth, cam_cfg["intrinsics"], is_ortho=cam_cfg["orthogonal"]
            )

            if (extrinsics := cam_cfg["extrinsics"]) is not None:
                # batched matrix-vector product
                # [..., 4, 4] @ [..., 3, 1] = [..., 3, 1]
                points = apply_homogeneous_transform(points, extrinsics)

            if cam_cfg["dynamic"]:
                dynamic_extrinsics, args = args[0], args[1:]
                assert dynamic_extrinsics.shape[-2:] == (4, 4)
                points = apply_homogeneous_transform(points, dynamic_extrinsics)

            # remove points that are beyond the max depth
            if self.max_depth is not None:
                mask = (depth < self.max_depth).flatten(start_dim=-2)
                points = points[mask]
            else:
                mask = Ellipsis

            all_points.append(points)

            if self.color:
                rgb, args = args[0], args[1:]
                # rgb and depth must have the same resolution
                assert rgb.shape[-3:] == cam_cfg["shape"]

                if cam_cfg["channel_order"] == "CHW":
                    # convert to HWC order
                    rgb = torch.movedim(rgb, -3, -1)

                rgb = rgb.flatten(start_dim=-3, end_dim=-2)

                # remove points that are beyond the max depth
                rgb = rgb[mask]
                all_rgb.append(rgb)

        # stack points from all cameras together
        # if we have batches of timesteps, they are kept separate
        all_points = torch.cat(all_points, dim=-2)

        assert len(args) == 0, f"Unused args: {args}"

        data = {"pos": all_points}

        if self.color:
            all_rgb = torch.cat(all_rgb, dim=-2)
            # x refers to "node features" in PyG
            data["x"] = all_rgb

        return Data(**data)
