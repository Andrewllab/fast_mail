from __future__ import annotations

from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import ReversibleTransform
from utils.math import normalize


class QuaternionRotations(ReversibleTransform):
    def __init__(self, specs: DataSpecs):
        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        return tensordict

    def reverse(self, tensordict: TensorDict) -> TensorDict:

        action = tensordict["action"]

        # normalize quaternions predicted by the model, which are likely not valid
        action[..., 3:7] = normalize(action[..., 3:7])

        return tensordict
