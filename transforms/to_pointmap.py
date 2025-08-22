import dataclasses
import torch
from tensordict import TensorDict

from environments.specs import (
    DataSpecs,
    PointMapStream,
    RGBStream,
    DepthStream,
    CameraSpec,
)
from transforms.base_transform import Transform

from utils.math import unproject_depth, transform_pointmap


class ToPointMap(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        color: bool = False,
        max_depth: float | None = None,
        out_key: str = "pointmap",
    ):
        super().__init__(specs=specs)
        self.color = color
        self.max_depth = max_depth
        self._out_key = out_key

        input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            and any(isinstance(stream, DepthStream) for stream in spec.streams.values())
        }
        self._input_specs = input_specs

        multiview = len(input_specs) > 1

        obs_specs = dict(specs.obs)
        for key, spec in input_specs.items():
            streams = dict(spec.streams)
            for name, stream in streams.items():
                if color and isinstance(stream, RGBStream):
                    streams.pop(name)  # Gets merged into pointmap later

                if not isinstance(stream, DepthStream):
                    continue

                if stream.intrinsics is None:
                    raise ValueError(
                        f"Depth stream at {key}.{name} does not have an intrinsics matrix."
                    )

                if color and not any(
                    isinstance(stream, RGBStream) for stream in spec.streams.values()
                ):
                    raise ValueError(
                        f"Depth camera spec {key} is not an RGBCameraSpec. Cannot use color."
                    )

                if multiview and spec.extrinsics is None:
                    raise ValueError(
                        f"Depth camera {key} does not have an extrinsics matrix."
                    )

                stream.pop(name)  # Remove depth stream from streams

                # Replace depth-stream with pointmap stream
                stream[self._out_key] = PointMapStream(
                    height=stream.height,
                    width=stream.width,
                    feature_dim=6 if color else 3,
                    color=color,
                    time=spec.time,
                    extrinsics=spec.extrinsics,
                )

            obs_specs[key] = dataclasses.replace(spec, streams=streams)

        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        for key, spec in self._input_specs.items():
            name, depth_stream = next(
                (name, stream)
                for name, stream in spec.streams.items()
                if isinstance(stream, DepthStream)
            )
            depth = tensordict.pop(("obs", key, name))

            assert depth_stream.intrinsics is not None, (
                "Depth stream must have intrinsics."
            )

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

                    # left multiply the dynamic extrinsics because they are
                    # applied after the static extrinsics
                    # (..., 4, 4) @ (4, 4) = (..., 4, 4)
                    extrinsics = dynamic_extrinsics @ extrinsics

                point_map = transform_pointmap(point_map, extrinsics)

            # Mask rgb map if max_depth is set
            rgb_name, rgb_stream = next(
                (
                    (name, stream)
                    for name, stream in spec.streams.items()
                    if isinstance(stream, RGBStream)
                ),
                (None, None),
            )

            if rgb_name is not None:
                rgb = tensordict["obs", key, rgb_name]

                if rgb_stream.channel_order == "CHW":
                    # convert to HWC order
                    rgb = torch.movedim(rgb, -3, -1)

                if self.max_depth is not None:
                    mask = depth > self.max_depth
                    rgb[mask] = 0

                if rgb.dtype == torch.uint8:
                    rgb = rgb.to(dtype=torch.get_default_dtype()).div(255)

                tensordict["obs", key, rgb_name] = rgb

            if self.color:
                assert rgb_name is not None
                tensordict.pop(
                    "obs", key, rgb_name
                )  # Remove RGB stream from tensordict

                # rgb and depth must have the same resolution
                # rgb: (..., H, W, 3)
                assert rgb.shape[-3:-1] == depth.shape[-2:], (
                    "RGB and depth must have the same spatial dimensions."
                )
                point_map = torch.cat([point_map, rgb], dim=-1)

            tensordict["obs", key, self._out_key] = point_map
        return tensordict
