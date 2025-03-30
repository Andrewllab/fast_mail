import dataclasses

import torch

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform


class ToTorchImage(Transform):
    def __init__(self, specs: DataSpecs) -> None:

        self._rgb_keys = [key for key, spec in specs.obs.items() if spec.type == "rgb"]
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key in self._rgb_keys:
            rgb_spec = obs_specs[key]
            # move last channel dimension from the end to the -3 position
            obs_specs[key] = dataclasses.replace(
                rgb_spec,
                shape=rgb_spec.shape[:-3] + rgb_spec.shape[-1:] + rgb_spec.shape[-3:-1],
            )
        self._specs = specs.replace(obs=obs_specs)

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
