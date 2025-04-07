import dataclasses

import torch
from tensordict import TensorDict

from environments.specs import CameraSpec, DataSpecs, RGBCameraSpec
from transforms.base_transform import Transform


class ToTorchImage(Transform):
    def __init__(self, specs: DataSpecs) -> None:

        # find the specs that this transform acts on
        input_specs = {
            key: spec for key, spec in specs.obs.items() if isinstance(spec, CameraSpec)
        }
        self._input_specs = input_specs

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

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        default_float_dtype = torch.get_default_dtype()

        for key, spec in self._input_specs.items():
            for subkey in spec.intensity_subkeys:
                nested_key = ("obs", key)
                if subkey is not None:
                    nested_key += (subkey,)
                image = tensordict[nested_key]

                image = image.to(dtype=default_float_dtype).div(255)

                tensordict[nested_key] = image

            if isinstance(spec, RGBCameraSpec):
                for subkey in spec.rgb_subkeys:
                    nested_key = ("obs", key)
                    if subkey is not None:
                        nested_key += (subkey,)
                    image = tensordict[nested_key]

                    if spec.channel_order == "HWC":
                        image = torch.movedim(image, -1, -3)

                    image = image.to(dtype=default_float_dtype).div(255)

                    tensordict[nested_key] = image

        return tensordict
