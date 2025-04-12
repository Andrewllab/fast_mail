from __future__ import annotations

import dataclasses

import torch
import torchvision.transforms.functional as F
from tensordict import TensorDict

from environments.specs import CameraSpec, DataSpecs, DepthCameraSpec, RGBCameraSpec
from transforms.base_transform import Transform


class CenterCropImage(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        shape: int | tuple[int, int] | None = None,
    ) -> None:

        # find the specs that this transform acts on
        input_specs = {
            key: spec for key, spec in specs.obs.items() if isinstance(spec, CameraSpec)
        }

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key, spec in input_specs.items():
            if isinstance(spec, RGBCameraSpec) and spec.channel_order != "CHW":
                spec = dataclasses.replace(
                    spec,
                    shape=spec.shape[:-3] + spec.shape[-1:] + spec.shape[-3:-1],
                    channel_order="CHW",
                )

            # compute new shape of image
            if isinstance(shape, int):
                new_shape = (shape, shape)
            elif isinstance(shape, tuple):
                # if shape is a tuple, it is the new shape
                new_shape = shape
            else:
                raise ValueError("Shape must be an int or a tuple of ints.")

            # update specs with resized shape and modified camera intrinsics
            spec = dataclasses.replace(
                spec,
                shape=(spec.shape[:-2] + new_shape),
            )
            if spec.intrinsics is not None:
                spec = dataclasses.replace(
                    spec,
                    intrinsics=spec.intrinsics.center_crop(new_shape),
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

            images = [(subkey, CameraSpec) for subkey in spec.intensity_subkeys]
            if isinstance(spec, RGBCameraSpec):
                images += [(subkey, RGBCameraSpec) for subkey in spec.rgb_subkeys]
            if isinstance(spec, DepthCameraSpec):
                images += [(subkey, DepthCameraSpec) for subkey in spec.depth_subkeys]

            for subkey, image_type in images:
                nested_key = ("obs", key)
                if subkey is not None:
                    nested_key += (subkey,)
                image = tensordict[nested_key]

                if image_type is RGBCameraSpec:
                    assert isinstance(spec, RGBCameraSpec)
                    if spec.channel_order == "HWC":
                        image = torch.movedim(image, -1, -3)

                    leading_dims = image.shape[:-3]
                    image = torch.flatten(image, end_dim=-4)
                else:
                    leading_dims = image.shape[:-2]
                    image = torch.flatten(image, end_dim=-3)

                image = F.center_crop(image, spec.shape[-2:])
                image = torch.unflatten(image, dim=0, sizes=leading_dims)

                tensordict[nested_key] = image

        return tensordict
