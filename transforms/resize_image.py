from __future__ import annotations

import dataclasses

import torch
import torchvision.transforms.functional as F
from tensordict import TensorDict
from torchvision.transforms import InterpolationMode

from environments.specs import CameraSpec, DataSpecs, DepthCameraSpec, RGBCameraSpec
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

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key, spec in input_specs.items():
            if isinstance(spec, RGBCameraSpec) and spec.channel_order != "CHW":
                raise ValueError(
                    f"Input spec {spec} must be in CHW format for normalization."
                )

            H, W = spec.shape[-2:]

            # compute new shape of image
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
            elif isinstance(shape, tuple):
                # if shape is a tuple, it is the new shape
                new_shape = shape
            else:
                raise ValueError("Shape must be an int or a tuple of ints, or None.")

            # update specs with resized shape and modified camera intrinsics
            spec = dataclasses.replace(
                spec,
                shape=(spec.shape[:-2] + new_shape),
            )
            if spec.intrinsics is not None:
                spec = dataclasses.replace(
                    spec,
                    intrinsics=spec.intrinsics.resize(new_shape),
                )
            obs_specs[key] = spec
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:

        for key, spec in self._output_specs.obs.items():
            if not isinstance(spec, CameraSpec):
                continue

            for subkey in spec.intensity_subkeys:
                nested_key = ("obs", key)
                if subkey is not None:
                    nested_key += (subkey,)
                image = tensordict[nested_key]

                leading_dims = image.shape[:-2]
                image = torch.flatten(image, end_dim=-3)
                image = F.resize(
                    image,
                    size=spec.shape[-2:],  # type: ignore
                    interpolation=self.interpolation,
                    antialias=self.antialias,
                )
                image = torch.unflatten(image, dim=0, sizes=leading_dims)

                tensordict[nested_key] = image

            if isinstance(spec, DepthCameraSpec):
                for subkey in spec.depth_subkeys:
                    nested_key = ("obs", key)
                    if subkey is not None:
                        nested_key += (subkey,)
                    image = tensordict[nested_key]

                    leading_dims = image.shape[:-2]
                    image = torch.flatten(image, end_dim=-3)
                    image = F.resize(
                        image,
                        size=spec.shape[-2:],  # type: ignore
                        interpolation=self.depth_interpolation,
                        antialias=self.antialias,
                    )
                    image = torch.unflatten(image, dim=0, sizes=leading_dims)

                    tensordict[nested_key] = image

            if isinstance(spec, RGBCameraSpec):
                for subkey in spec.rgb_subkeys:
                    nested_key = ("obs", key)
                    if subkey is not None:
                        nested_key += (subkey,)
                    image = tensordict[nested_key]

                    leading_dims = image.shape[:-3]
                    image = torch.flatten(image, end_dim=-4)
                    image = F.resize(
                        image,
                        size=spec.shape[-2:],  # type: ignore
                        interpolation=self.interpolation,
                        antialias=self.antialias,
                    )
                    image = torch.unflatten(image, dim=0, sizes=leading_dims)

                    tensordict[nested_key] = image

        return tensordict
