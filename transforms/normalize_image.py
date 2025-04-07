import torchvision.transforms.functional as F

from environments.specs import (
    DataSpecs,
    RGBCameraSpec,
    RGBDCameraSpec,
    StereoRGBCameraSpec,
)
from transforms.base_transform import KeyMapping, Transform


class NormalizeImage(Transform):
    def __init__(self, specs: DataSpecs, mean, std) -> None:

        self.mean = mean
        self.std = std
        self._input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, RGBCameraSpec)
        }
        self._output_specs = specs  # no changes to specs

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
        return F.normalize(image, self.mean, self.std, inplace=True)
