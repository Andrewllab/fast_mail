import dataclasses
import logging

import torch
from tensordict import TensorDict

from environments.specs import (
    CameraSpec,
    DataSpecs,
    DepthStream,
    PointMapStream,
    RGBStream,
)
from transforms.base_transform import Transform
from utils.math import transform_pointmap, unproject_depth

log = logging.getLogger(__name__)


class ToPointMap(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        color: bool = False,
        max_depth: float | None = None,
        out_key: str = "pointmap",
    ):
        self.color = color
        self.max_depth = max_depth
        self._out_key = out_key

        depth_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            and any(isinstance(stream, DepthStream) for stream in spec.streams.values())
        }

        multiview = len(depth_specs) > 1

        obs_specs = dict(specs.obs)
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

            if color:
                rgb_streams = [
                    name
                    for name, stream in spec.streams.items()
                    if isinstance(stream, RGBStream)
                ]

                if not rgb_streams:
                    raise ValueError(
                        f"Camera spec {key} is not an RGBCameraSpec. Cannot use color."
                    )
                elif len(rgb_streams) > 1:
                    log.warning(
                        f"Camera spec '{key}' contains multiple RGB streams. "
                        f"Only the first RGB stream {rgb_streams[0]} will be concatenated "
                        "to the pointmap feature channel"
                    )

                rgb_name = rgb_streams[0]

            if multiview and spec.extrinsics is None:
                raise ValueError(
                    f"Depth camera {key} does not have an extrinsics matrix."
                )
            if spec.dynamic_pose_obs_key is not None and spec.extrinsics is None:
                raise ValueError(
                    f"Dynamic pose obs key {spec.dynamic_pose_obs_key} is not supported for depth cameras without extrinsics."
                )

            streams = dict(spec.streams)
            streams.pop(name)  # Remove depth stream from streams
            if color:
                streams.pop(rgb_name)  # Remove RGB stream from streams

            # Replace depth-stream with pointmap stream
            streams[self._out_key] = PointMapStream(
                height=depth_stream.height,
                width=depth_stream.width,
                channels=6 if color else 3,
                color=color,
                time=spec.time,
                intrinsics=depth_stream.intrinsics,
                extrinsics=spec.extrinsics,
            )

            obs_specs[key] = dataclasses.replace(spec, streams=streams)

        self._input_specs = depth_specs
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        default_float_dtype = torch.get_default_dtype()

        for key, spec in self._input_specs.items():

            name, depth_stream = next(
                (name, stream)
                for name, stream in spec.streams.items()
                if isinstance(stream, DepthStream)
            )
            depth = tensordict.pop(("obs", key, name))

            assert depth_stream.intrinsics is not None
            # points: (..., H, W, 3)
            point_map = unproject_depth(
                depth,
                depth_stream.intrinsics.intrinsic_matrix.to(depth.device),
                is_ortho=depth_stream.orthogonal,
            )
            if (extrinsics := spec.extrinsics) is not None:
                extrinsics = extrinsics.to(point_map.device)

                if (pose_key := spec.dynamic_pose_obs_key) is not None:
                    if not isinstance(pose_key, tuple):
                        pose_key = (pose_key,)
                    dynamic_extrinsics = tensordict[("obs",) + pose_key]
                    assert dynamic_extrinsics.shape[-2:] == (4, 4)
                    dynamic_extrinsics = dynamic_extrinsics.to(
                        dtype=torch.float32
                    )  # BackCompat

                    # left multiply the dynamic extrinsics because they are
                    # applied after the static extrinsics
                    # (..., 4, 4) @ (4, 4) = (..., 4, 4)
                    extrinsics = dynamic_extrinsics @ extrinsics

                point_map = transform_pointmap(point_map, extrinsics)

            if self.color:
                name, rgb_stream = next(
                    (name, stream)
                    for name, stream in spec.streams.items()
                    if isinstance(stream, RGBStream)
                )
                rgb = tensordict.pop(("obs", key, name))

                if rgb_stream.channel_order == "CHW":
                    # convert to HWC order
                    rgb = torch.movedim(rgb, -3, -1)

                if rgb.dtype == torch.uint8:
                    rgb = rgb.to(dtype=default_float_dtype).div(255)

                # rgb and point_map must have the same resolution
                # rgb: (..., H, W, 3)
                point_map = torch.cat([point_map, rgb], dim=-1)

            # remove points that are beyond the max depth
            if self.max_depth is not None:
                # get mask of points that are within max depth
                # mask: (..., H, W)
                mask = torch.logical_or(depth < 0, depth >= self.max_depth)
                point_map[mask] = 0

            tensordict["obs", key, self._out_key] = point_map

        return tensordict
