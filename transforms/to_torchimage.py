import dataclasses

import torch

from environments.specs import (
    DataSpecs,
    RGBCameraSpec,
    RGBDCameraSpec,
    StereoRGBCameraSpec,
)
from transforms.base_transform import KeyMapping, Transform


class ToTorchImage(Transform):
    def __init__(self, specs: DataSpecs) -> None:

        self._input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, RGBCameraSpec)
        }

        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key in self._input_specs:
            rgb_spec = obs_specs[key]
            assert isinstance(rgb_spec, RGBCameraSpec)
            if rgb_spec.channel_order == "HWC":
                # move last channel dimension from the end to the -3 position
                obs_specs[key] = dataclasses.replace(
                    rgb_spec,
                    shape=rgb_spec.shape[:-3]
                    + rgb_spec.shape[-1:]
                    + rgb_spec.shape[-3:-1],
                )
        self._output_specs = specs.replace(obs=obs_specs)

        nested_keys = []
        for key, spec in self._input_specs.items():
            if isinstance(spec, RGBDCameraSpec):
                nested_keys.append(("obs", key, "rgb"))
            elif isinstance(spec, StereoRGBCameraSpec):
                nested_keys.extend([("obs", key, "left"), ("obs", key, "right")])
            else:
                nested_keys.append(("obs", key))

        self._nested_keys = nested_keys

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [KeyMapping(in_keys=keys, out_keys=keys) for keys in self._nested_keys]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, image):
        in_key = get_in_keys()[0]
        key_idx = self._nested_keys.index(in_key)
        input_spec = list(self._input_specs.values())[key_idx]

        default_float_dtype = torch.get_default_dtype()

        if input_spec.channel_order == "HWC":
            # if the input is in HWC format, we need to move the last channel dimension
            # to the -3 position
            image = torch.movedim(image, -1, -3)

        # convert to (some sort of) float and rescale to [0, 1]
        return image.to(dtype=default_float_dtype).div(255)


import inspect

from transforms.base_transform import KeyType


def get_in_keys() -> KeyType:
    this_frame = inspect.currentframe()
    td_frame = this_frame.f_back.f_back
    td_module: TensorDictModule = td_frame.f_locals["self"]
    return td_module.in_keys
