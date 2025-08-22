from __future__ import annotations

import torch
import torchvision.transforms.functional as F
from tensordict import TensorDict
from torchvision.transforms import InterpolationMode

from environments.specs import CameraSpec, DataSpecs, DepthStream
from transforms.base_transform import Transform


class ResizeImage(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        factor: float | None = None,
        shape: int | tuple[int, int] | None = None,
        interpolation: InterpolationMode | str | None = None,
        depth_interpolation: InterpolationMode | str | None = None,
        antialias: bool = True,
    ) -> None:

        if factor is None and shape is None:
            raise ValueError("Either factor or shape must be provided.")
        elif factor is not None and shape is not None:
            raise ValueError("Only one of factor or shape may be provided.")

        if interpolation is None:
            interpolation = InterpolationMode.BILINEAR
        elif isinstance(interpolation, str):
            interpolation = InterpolationMode[interpolation]
        self.interpolation = interpolation

        if depth_interpolation is None:
            depth_interpolation = InterpolationMode.NEAREST_EXACT
        elif isinstance(depth_interpolation, str):
            depth_interpolation = InterpolationMode[depth_interpolation]
        self.depth_interpolation = depth_interpolation

        self.antialias = antialias

        # find the specs that this transform acts on
        input_specs = {
            key: spec for key, spec in specs.obs.items() if isinstance(spec, CameraSpec)
        }
        self._input_specs = input_specs

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key, spec in input_specs.items():
            streams = dict(spec.streams)  # copy streams for local modification
            for name, stream in streams.items():
                stream = stream.reorder_channels("CHW")

                # compute new shape of image
                H, W = stream.height_width
                if factor is not None:
                    new_shape = (int(factor * H), int(factor * W))
                elif isinstance(shape, int):
                    # if shape is an int, it is the new size for the shortest side,
                    # and the longest side is scaled to maintain the aspect ratio
                    shortest_side = min(H, W)
                    new_shape = (
                        int(shape * H / shortest_side),
                        int(shape * W / shortest_side),
                    )
                elif isinstance(shape, (tuple, list)):
                    # if shape is a tuple, it is the new shape
                    new_shape = tuple(shape)
                else:
                    raise ValueError(
                        "Shape must be an int or a tuple of ints, or None."
                    )

                stream = stream.resize(new_shape)

                # update stream with resized shape and modified camera intrinsics
                streams[name] = stream

            obs_specs[key] = spec.replace(streams=streams)
        self._output_specs = specs.replace(obs=obs_specs)

        if factor is not None:
            self.summary = f"(factor={factor})"
        elif isinstance(shape, int):
            self.summary = f"(shape={shape}x{shape})"
        elif isinstance(shape, (tuple, list)):
            self.summary = f"(shape={shape[0]}x{shape[1]})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.summary}"

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        default_float_dtype = torch.get_default_dtype()

        for key, spec in self._input_specs.items():
            output_spec = self._output_specs.obs[key]
            assert isinstance(output_spec, CameraSpec)

            images = tensordict["obs", key]
            for name, stream in spec.streams.items():
                output_stream = output_spec.streams[name]

                image = images[name]

                n_image_dims = stream.n_image_dims
                leading_dims = image.shape[:-n_image_dims]
                image = torch.flatten(image, end_dim=-n_image_dims - 1)

                if stream.channel_order == "HWC":
                    image = torch.movedim(image, -1, -3)

                if image.dtype != default_float_dtype:
                    image = image.to(dtype=default_float_dtype).div(255)

                interpolation = (
                    self.depth_interpolation
                    if isinstance(stream, DepthStream)
                    else self.interpolation
                )

                image = F.resize(
                    image,
                    size=output_stream.height_width,  # type: ignore
                    interpolation=interpolation,
                    antialias=self.antialias,
                )
                image = torch.unflatten(image, dim=0, sizes=leading_dims)

                images[name] = image

        return tensordict
