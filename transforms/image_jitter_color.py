from __future__ import annotations

from typing import Union

import torch
from tensordict import TensorDict
from torchvision.transforms.v2 import ColorJitter

from environments.specs import CameraSpec, DataSpecs, RGBStream
from transforms.base_transform import Transform


class ColorJitterImage(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        brightness: Union[float, tuple[float, float]],
        contrast: Union[float, tuple[float, float]],
        saturation: Union[float, tuple[float, float]],
        hue: Union[float, tuple[float, float]],
    ) -> None:

        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        # find the specs that this transform acts on
        input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            and any(isinstance(stream, RGBStream) for stream in spec.streams.values())
        }
        self._input_specs = input_specs

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key, spec in input_specs.items():
            streams = dict(spec.streams)  # copy streams for local modification
            for name, stream in streams.items():
                if not isinstance(stream, RGBStream):
                    continue
                stream = stream.reorder_channels("CHW")
                streams[name] = stream
            obs_specs[key] = spec.replace(streams=streams)
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if not self.training:
            return tensordict
        
        default_float_dtype = torch.get_default_dtype()

        for key, spec in self._input_specs.items():
            images = tensordict["obs", key]
            for name, stream in spec.streams.items():
                if not isinstance(stream, RGBStream):
                    continue

                image = images[name]

                if stream.channel_order == "HWC":
                    image = torch.movedim(image, -1, -3)

                if image.dtype == torch.uint8:
                    image = image.to(dtype=default_float_dtype).div(255)

                image = ColorJitter(
                    brightness=self.brightness,
                    contrast=self.contrast,
                    saturation=self.saturation,
                    hue=self.hue,
                )(image)

                images[name] = image

        return tensordict
