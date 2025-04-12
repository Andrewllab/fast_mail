import dataclasses

import torch
import torchvision.transforms.functional as F

from environments.specs import DataSpecs, RGBCameraSpec
from transforms.base_transform import KeyMapping, Transform


class NormalizeImage(Transform):
    def __init__(self, specs: DataSpecs, mean, std) -> None:

        self.mean = mean
        self.std = std

        # find the specs that this transform acts on
        input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, RGBCameraSpec)
        }

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        for key, spec in input_specs.items():
            if spec.channel_order != "CHW":
                spec = dataclasses.replace(
                    spec,
                    shape=spec.shape[:-3] + spec.shape[-1:] + spec.shape[-3:-1],
                    channel_order="CHW",
                )
            obs_specs[key] = spec
        self._output_specs = specs.replace(obs=obs_specs)

        # create a list of key mappings for the forward call
        key_mappings = []
        for key, spec in input_specs.items():
            for subkey in spec.rgb_subkeys:
                nested_key = ("obs", key)
                if subkey is not None:
                    nested_key += (subkey,)
                key_mappings.append(KeyMapping(in_keys=nested_key, out_keys=nested_key))
        self._key_mappings = key_mappings

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return self._key_mappings

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(self, image: torch.Tensor) -> torch.Tensor:
        default_float_dtype = torch.get_default_dtype()

        if image.shape[-1] == 3:
            image = torch.movedim(image, -1, -3)
        if image.dtype != default_float_dtype:
            image = image.to(dtype=default_float_dtype).div(255)

        return F.normalize(image, self.mean, self.std, inplace=True)
