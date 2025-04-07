import dataclasses

import torch

from environments.specs import DataSpecs, IntensityCameraSpec, RGBCameraSpec
from transforms.base_transform import KeyMapping, Transform


class ToTorchImage(Transform):
    def __init__(self, specs: DataSpecs) -> None:

        # find the specs that this transform acts on
        input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, IntensityCameraSpec)
        }

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key, spec in input_specs.items():
            if isinstance(spec, RGBCameraSpec) and spec.channel_order == "HWC":
                # move last channel dimension from the end to the -3 position
                obs_specs[key] = dataclasses.replace(
                    spec,
                    shape=spec.shape[:-3] + spec.shape[-1:] + spec.shape[-3:-1],
                    channel_order="CHW",
                )
        self._output_specs = specs.replace(obs=obs_specs)

        # create a list of key mappings for the forward call
        key_mappings = []
        for key, spec in input_specs.items():
            for subkey in spec.image_subkeys:
                if subkey is not None:
                    nested_key = ("obs", key, subkey)
                else:
                    nested_key = ("obs", key)
                key_mappings.append(
                    KeyMapping(
                        in_keys=nested_key,
                        out_keys=nested_key,
                        args=(spec,),
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
        self, image: torch.Tensor, input_spec: IntensityCameraSpec
    ) -> torch.Tensor:
        default_float_dtype = torch.get_default_dtype()

        if isinstance(input_spec, RGBCameraSpec) and input_spec.channel_order == "HWC":
            # if the input is in HWC format, we need to move the last channel dimension
            # to the -3 position
            image = torch.movedim(image, -1, -3)

        # convert to (some sort of) float and rescale to [0, 1]
        return image.to(dtype=default_float_dtype).div(255)
