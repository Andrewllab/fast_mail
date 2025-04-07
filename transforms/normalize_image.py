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
        for spec in input_specs.values():
            if spec.channel_order != "CHW":
                raise ValueError(
                    f"Input spec {spec} must be in CHW format for normalization."
                )

        self._output_specs = specs  # no changes to specs

        # create a list of key mappings for the forward call
        key_mappings = []
        for key, spec in input_specs.items():
            for subkey in spec.rgb_subkeys:
                if subkey is not None:
                    nested_key = ("obs", key, subkey)
                else:
                    nested_key = ("obs", key)
                key_mappings.append(KeyMapping(in_keys=nested_key, out_keys=nested_key))
        self._key_mappings = key_mappings

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return self._key_mappings

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(self, image):
        return F.normalize(image, self.mean, self.std, inplace=True)
