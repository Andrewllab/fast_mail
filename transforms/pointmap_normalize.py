from __future__ import annotations

import torch
import torch.nn as nn

from environments.specs import DataSpecs, PointMapStream, CameraSpec
from transforms.base_transform import NormalizingTransform


class PointMapNormalize(NormalizingTransform, nn.Module):
    max_points: torch.Tensor
    min_points: torch.Tensor
    
    def __init__(self, specs: DataSpecs, do_reverse: bool = False):
        super().__init__()

        self._specs = specs
        self.do_reverse = do_reverse

        self._input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            and any(isinstance(stream, PointMapStream) for stream in spec.streams.values())
        }
        
        streams = [
            stream
            for spec in self._input_specs.values()
            for stream in spec.streams.values()
            if isinstance(stream, PointMapStream)
        ]
        if not streams:
            raise ValueError("No PointMapStream found in specs")
        
        channels = [stream.channels for stream in streams]
        if not all(c == channels[0] for c in channels):
            raise ValueError(
                f"All camera streams must have the same number of channels, but got {channels}"
            )
            
        if channels[0] != 3:
            raise ValueError(f"Currently only support 3-channel pointmaps, got {channels[0]}")

        self.register_buffer("max_points", torch.ones(channels[0]) * float('-inf'))
        self.register_buffer("min_points", torch.ones(channels[0]) * float('inf'))

    @property
    def specs(self) -> DataSpecs:
        return self._specs
    
    def call_trajectory(self, tensordict):
        """Collect statistics over a trajectory of pointmaps to compute mean and std."""        
        for key, spec in self._input_specs.items():
            for name, stream in spec.streams.items():
                if not isinstance(stream, PointMapStream):
                    continue

                pointmap: torch.Tensor = tensordict["obs", key][name]
                
                # Move channels to HWC
                if stream.channel_order == "CHW":
                    pointmap = torch.movedim(pointmap, -3, -1)

                # (..., H, W, C) -> (N, C)
                pointmap = pointmap.flatten(0, -2)
                assert pointmap.dim() == 2
                assert pointmap.shape[-1] == 3

                # Collect max and min values
                max_points = pointmap.max(dim=0).values
                min_points = pointmap.min(dim=0).values
                self.max_points = torch.maximum(self.max_points, max_points)
                self.min_points = torch.minimum(self.min_points, min_points)

        print(self.max_points)
        print(self.min_points)
        return tensordict
    
    def forward(self, tensordict):
        if any(torch.isinf(self.max_points)) or any(torch.isinf(self.min_points)):
            raise ValueError(
                "Found inf in max_points or min_points. Make sure to call call_trajectory on a representative dataset before using the transform."
            )
            
        if any(self.max_points <= self.min_points):
            raise ValueError(
                "max_points must be greater than min_points for all channels."
            )

        for key, spec in self._input_specs.items():
            pointmaps = tensordict["obs", key]
            for name, stream in spec.streams.items():
                if not isinstance(stream, PointMapStream):
                    continue

                pointmap = pointmaps[name]

                # Scale pointmap from [min_points, max_points] to [-1, 1] on each axis
                pointmap = 2 * (pointmap - self.min_points) / (self.max_points - self.min_points) - 1.0
                pointmap = torch.clamp(pointmap, -1.0, 1.0)
                    
                pointmaps[name] = pointmap

        return tensordict
    
    def reverse(self, tensordict):
        if not self.do_reverse:
            return tensordict

        for key, spec in self._input_specs.items():
            pointmaps = tensordict["obs", key]
            for name, stream in spec.streams.items():
                if not isinstance(stream, PointMapStream):
                    continue
                
                pointmap: torch.Tensor = pointmaps[name]
                
                # Unscale pointmap from [-1, 1] to [min_points, max_points] on each axis
                pointmap = torch.clamp(pointmap, -1.0, 1.0)
                pointmap = ((pointmap + 1.0) / 2) * (self.max_points - self.min_points) + self.min_points
                    
                pointmaps[name] = pointmap
                
        return tensordict