from __future__ import annotations
from typing import Sequence, Union

import torch
from tensordict import TensorDict

from environments.specs import CameraSpec, DataSpecs, DepthStream
from transforms.base_transform import Transform

# In signal processing theory, Gaussian noise, named after Carl Friedrich Gauss,
#  is a kind of signal noise that has a probability density function (pdf) equal to that of the normal distribution (which is also known as the Gaussian distribution).
# https://en.wikipedia.org/wiki/Gaussian_noise


class JitterDepthMap(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        sigma: Union[float, int, Sequence[Union[float, int]]],
    ) -> None:

        self.sigma = sigma

        # find the specs that this transform acts on
        input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            and any(isinstance(stream, DepthStream) for stream in spec.streams.values())
        }
        self._input_specs = input_specs

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key, spec in input_specs.items():
            streams = dict(spec.streams)  # copy streams for local modification
            for name, stream in streams.items():
                stream = stream.reorder_channels("HW")
                streams[name] = stream
            obs_specs[key] = spec.replace(streams=streams)
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        default_float_dtype = torch.get_default_dtype()

        for key, spec in self._input_specs.items():
            depth_maps = tensordict["obs", key]
            for name, stream in spec.streams.items():
                if not isinstance(stream, DepthStream):
                    continue

                depth_map = depth_maps[name]

                if depth_map.dtype != default_float_dtype:
                    depth_map = depth_map.to(dtype=default_float_dtype)

                depth_map += (
                    torch.randn_like(depth_map) * self.sigma
                )  # apply gaussian noise

                depth_map = torch.clamp(depth_map, min=0.0)

                depth_maps[name] = depth_map

        return tensordict
