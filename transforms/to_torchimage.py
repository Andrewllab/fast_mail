import dataclasses

import torch

from environments.specs import DataSpecs, RGBCameraSpec
from transforms.base_transform import KeyMapping, Transform


class ToTorchImage(Transform):
    def __init__(self, specs: DataSpecs) -> None:

        # find the specs that this transform acts on
        input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, RGBCameraSpec)
        }

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key, rgb_spec in input_specs.items():
            if rgb_spec.channel_order == "HWC":
                # move last channel dimension from the end to the -3 position
                obs_specs[key] = dataclasses.replace(
                    rgb_spec,
                    shape=rgb_spec.shape[:-3]
                    + rgb_spec.shape[-1:]
                    + rgb_spec.shape[-3:-1],
                    channel_order="CHW",
                )
        self._output_specs = specs.replace(obs=obs_specs)

        # create a list of key mappings for the forward call
        nested_keys = []
        for key, spec in input_specs.items():
            for subkey in spec.rgb_subkeys:
                if subkey is not None:
                    nested_keys.append(("obs", key, subkey))
                else:
                    nested_keys.append(("obs", key))
        self._nested_keys = nested_keys

        # store the input specs so we can use them in the call method
        self._input_specs = list(input_specs.values())

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [KeyMapping(in_keys=keys, out_keys=keys) for keys in self._nested_keys]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(self, image):
        input_spec = self._input_specs[self.mapping_idx]

        default_float_dtype = torch.get_default_dtype()

        if input_spec.channel_order == "HWC":
            # if the input is in HWC format, we need to move the last channel dimension
            # to the -3 position
            image = torch.movedim(image, -1, -3)

        # convert to (some sort of) float and rescale to [0, 1]
        return image.to(dtype=default_float_dtype).div(255)
