from __future__ import annotations

import logging

from tensordict import TensorDict
from torch import Tensor

from environments.specs import CameraSpec, DataSpecs, DepthStream
from transforms.base_transform import Transform

log = logging.getLogger(__name__)


class ClampDepth(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        max: float,
    ):
        self.max = max
        self._output_specs = specs

        self.streams = {
            (key, name): stream
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            for name, stream in spec.streams.items()
            if isinstance(stream, DepthStream)
        }

        if not self.streams:
            log.warning("No depth streams found in input specs.")

    @property
    def specs(self):
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        for key, name in self.streams.keys():
            depth: Tensor = tensordict["obs", key, name]

            # assume that nan values are the same as infinity, which would be
            # clamped in the next step
            depth.nan_to_num_(nan=self.max)

            # also clamp negative depths to zero
            depth.clamp_(min=0.0, max=self.max)

        return tensordict

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max={self.max})"
