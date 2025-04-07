from __future__ import annotations

import dataclasses

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from environments.specs import DataSpecs, IntensityCameraSpec, RGBCameraSpec
from transforms.base_transform import KeyMapping, Transform


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
            interpolation = InterpolationMode(interpolation)

        if depth_interpolation is None:
            depth_interpolation = InterpolationMode.NEAREST_EXACT
        elif isinstance(depth_interpolation, str):
            depth_interpolation = InterpolationMode(depth_interpolation)

        self.antialias = antialias

        # find the specs that this transform acts on
        input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, IntensityCameraSpec)
        }

        # create a dictionary of kwargs for resize() for each camera image, and
        # create a modified specs object for the output
        cam_kwargs = []
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

            # save the kwargs for resize()
            cam_kwargs.append(
                {
                    "size": new_shape,
                    "interpolation": (
                        interpolation
                        if isinstance(spec, RGBCameraSpec)
                        else depth_interpolation
                    ),
                }
            )

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
        # self.cam_kwargs = cam_kwargs
        self._output_specs = specs.replace(obs=obs_specs)

        # create a list of key mappings for the forward call
        key_mappings = []
        for (key, spec), kwargs in zip(input_specs.items(), cam_kwargs):
            for subkey in spec.subkeys:
                if subkey is not None:
                    nested_key = ("obs", key, subkey)
                else:
                    nested_key = ("obs", key)
                key_mappings.append(
                    KeyMapping(
                        in_keys=nested_key,
                        out_keys=nested_key,
                        args=(kwargs, spec),
                    )
                )
        self._key_mappings = key_mappings

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return self._key_mappings

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(
        self, image: torch.Tensor, kwargs: dict, input_spec: IntensityCameraSpec
    ) -> torch.Tensor:
        if isinstance(input_spec, RGBCameraSpec):
            n_image_dim = 3
        else:
            n_image_dim = 2

        leading_dims = image.shape[:-n_image_dim]
        image = torch.flatten(image, end_dim=-n_image_dim - 1)
        image = F.resize(image, **kwargs, antialias=self.antialias)
        image = torch.unflatten(image, dim=0, sizes=leading_dims)

        return image
