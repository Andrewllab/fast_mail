import dataclasses

import torch

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform


class ToTorchImage(Transform):
    def __init__(self, specs: DataSpecs) -> None:

        self._obs_spec = dict(specs.obs)  # copy obs spec for local modification
        self._rgb_keys = [key for key, spec in specs.obs.items() if spec.type == "rgb"]
        for key in self._rgb_keys:
            spec = self._obs_spec[key]
            # move last channel dimension from the end to the -3 position
            self._obs_spec[key] = dataclasses.replace(
                spec, shape=spec.shape[:-3] + spec.shape[-1:] + spec.shape[-3:-1]
            )
        self._specs = DataSpecs(_obs=self._obs_spec, action=specs.action)

    @property
    def key_mappings(self):
        return [
            KeyMapping(in_keys=[("obs", key)], out_keys=[("obs", key)])
            for key in self._rgb_keys
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, image):
        default_float_dtype = torch.get_default_dtype()

        return (
            torch.movedim(image, -1, -3)  # put it from HWC to CHW format
            .to(dtype=default_float_dtype)  # convert to (some sort of) float
            .div(255)  # rescale to between 0.0 and 1.0
        )
