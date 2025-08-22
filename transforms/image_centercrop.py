from __future__ import annotations

import dataclasses

import torch
import torchvision.transforms.functional as F
from tensordict import TensorDict

from environments.specs import CameraSpec, DataSpecs
from transforms.base_transform import Transform


class CenterCropImage(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        shape: int | tuple[int, int],
    ) -> None:

        # compute new shape of image
        if isinstance(shape, int):
            new_shape = (shape, shape)
        elif isinstance(shape, tuple):
            # if shape is a tuple, it is the new shape
            new_shape = shape
        else:
            raise ValueError("Shape must be an int or a tuple of ints.")

        # find the specs that this transform acts on
        input_specs = {
            key: spec for key, spec in specs.obs.items() if isinstance(spec, CameraSpec)
        }

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key, spec in input_specs.items():
            streams = dict(spec.streams)  # copy streams for local modification
            for name, stream in streams.items():
                stream = stream.reorder_channels("CHW")
                stream = stream.center_crop(new_shape)

                # update stream with resized shape and modified camera intrinsics
                streams[name] = stream

            obs_specs[key] = dataclasses.replace(spec, streams=streams)
        self._output_specs = specs.replace(obs=obs_specs)

        self.summary = f"(shape={new_shape})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.summary}"

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:

        for key, spec in self._output_specs.obs.items():
            if not isinstance(spec, CameraSpec):
                continue

            images = tensordict["obs", key]
            for name, stream in spec.streams.items():

                image = images[name]

                n_image_dims = stream.n_image_dims
                leading_dims = image.shape[:-n_image_dims]
                image = torch.flatten(image, end_dim=-n_image_dims - 1)

                if stream.channel_order == "HWC":
                    image = torch.movedim(image, -1, -3)

                image = F.center_crop(image, stream.height_width)
                image = torch.unflatten(image, dim=0, sizes=leading_dims)

                images[name] = image

        return tensordict
