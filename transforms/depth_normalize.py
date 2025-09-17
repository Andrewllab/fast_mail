from __future__ import annotations

import logging

import torch
from tensordict import TensorDict

from environments.specs import DataSpecs, CameraSpec, DepthStream
from transforms.base_transform import Transform

log = logging.getLogger(__name__)


class DepthNormalize(Transform):

    def __init__(
        self, 
        specs: DataSpecs,
        max_depth: float,
    ):
        super().__init__()

        self._input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec) and any(isinstance(s, DepthStream) for s in spec.streams.values())
        }

        if not self._input_specs:
            raise ValueError("No depth streams found in the provided specs.")
        
        self._specs = specs

        self.max_depth = max_depth
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive.")
        
    @property
    def specs(self) -> DataSpecs:
        return self._specs
            
    def __call__(self, tensordict: TensorDict) -> TensorDict:
            
        for key, spec in self._input_specs.items():
            for name, stream in spec.streams.items():
                if not isinstance(stream, DepthStream):
                    continue
            
                # Scale depth map from [0, max_depth] to [-1, 1]
                depth_map: torch.Tensor = tensordict["obs", key][name]
                depth_map = torch.clamp(depth_map, 0.0, self.max_depth)
                depth_map = 2 * (depth_map / self.max_depth) - 1.0
                depth_map = torch.clamp(depth_map, -1.0, 1.0)
                tensordict["obs", key][name] = depth_map

        return tensordict