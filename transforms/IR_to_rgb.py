import torchvision.transforms.functional as F
import numpy as np

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform


class ResizeImage(Transform):
    def __init__(self, specs: DataSpecs) -> None:

        self._ir_keys = [key for key, spec in specs.obs.items() if spec.type == "ir"] # TODO: Handle IR stereo obs correctly: { "ir": { "left": img1, "right": img2 } } ?
        self._specs = specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=[("ir", key)],
                out_keys=[("ir", key)])
            for key in self._ir_keys
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, image):
        return np.stack([image] * 3, axis=-1)
