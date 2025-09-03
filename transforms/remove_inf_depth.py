from __future__ import annotations

import logging

import torch
from tensordict import TensorDict

from environments.specs import CameraSpec, DataSpecs, DepthStream, EmbedSpec, RGBStream
from transforms.base_transform import Transform

log = logging.getLogger(__name__)

class RemoveInfDepthStream(Transform):
    def __init__(
        self,
        specs: DataSpecs,

    ):
        self._output_specs = specs
        self._has_depth_stream = any(
            isinstance(stream, DepthStream)
            for _, cam_spec in specs.obs.items()
            if isinstance(cam_spec, CameraSpec)
            for _, stream in cam_spec.streams.items()
        )
        self._imgs_transformed = 0
        
        if not self._has_depth_stream:
            log.warning("No depth stream found in input specs.")

    @property
    def specs(self):
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if not self._has_depth_stream:
            return tensordict

        for key, spec in self._output_specs.obs.items():
            if not isinstance(spec, CameraSpec):
                continue
            for stream_name, stream in spec.streams.items():
                if not isinstance(stream, DepthStream):
                    continue
                depth = tensordict["obs", key][stream_name]
                if not torch.isfinite(depth).all():
                    depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
                    tensordict["obs", key][stream_name] = depth
                    log.info(f"Removed inf/nan values from depth stream in {key}.")
        return tensordict