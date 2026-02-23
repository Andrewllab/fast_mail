from __future__ import annotations

import logging

import torch
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import Transform

log = logging.getLogger(__name__)


class NoOpTransform(Transform):

    def __init__(
        self,
        specs: DataSpecs,
    ):
        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        return tensordict
