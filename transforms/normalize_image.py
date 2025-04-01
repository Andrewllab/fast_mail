import torchvision.transforms.functional as F

from environments.specs import CameraSpec, DataSpecs
from transforms.base_transform import KeyMapping, Transform


class NormalizeImage(Transform):
    def __init__(self, specs: DataSpecs, mean, std) -> None:

        self.mean = mean
        self.std = std
        self._rgb_keys = [
            key for key, spec in specs.obs.items() if type(spec) == CameraSpec
        ]
        self._specs = specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(in_keys=[("obs", key)], out_keys=[("obs", key)])
            for key in self._rgb_keys
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, image):
        return F.normalize(image, self.mean, self.std, inplace=True)
