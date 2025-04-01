import torchvision.transforms.functional as F
import numpy as np

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform


class IRToRGB(Transform):
    def __init__(self, specs: DataSpecs) -> None:

        self._ir_keys = [key for key, spec in specs.obs.items() if spec.type == "ir"] # TODO: Assume that the data is stored like: { "ir": { "left": img1, "right": img2 } } ?
        self._specs = specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=[("obs", key)],
                out_keys=[("obs", key)])
            for key in self._ir_keys
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, ir_obs):
        for side in ir_obs:
            ir_obs[side] = np.stack(ir_obs[side] * 3, axis=-1)
        return ir_obs
