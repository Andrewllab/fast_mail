from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
import torchvision.transforms.functional as F
import rootutils

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from environments.specs import DataSpecs, PointMapStream, CameraSpec
from transforms.base_transform import NormalizingTransform


class PointMapNormalize(NormalizingTransform, nn.Module):
    def __init__(self, specs: DataSpecs):
        super().__init__()

        self._specs = specs
        
        self._input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            and any(isinstance(stream, PointMapStream) for stream in spec.streams.values())
        }
        
        streams = [
            stream
            for spec in specs.obs.values()
            for _, stream in spec.streams.items()
            if isinstance(stream, PointMapStream)
        ]
        if not streams:
            raise ValueError("No PointMapStream found in specs")
        
        H, W = None, None
        n_channels = None
        for stream in streams:
            if H is None or W is None:
                H, W = stream.height_width
            else:
                assert stream.height_width == (H, W), "All PointMapStreams must have the same spatial dimensions"

            if n_channels is None:
                n_channels = stream.channels
            else:
                assert stream.channels == n_channels, "All PointMapStreams must have the same number of channels"

        assert n_channels is not None
        self.register_buffer("mean", torch.zeros(n_channels))
        self.register_buffer("m2", torch.zeros(n_channels))
        self.n_points = 0
        
    @property
    def specs(self) -> DataSpecs:
        return self._specs
    
    def call_trajectory(self, tensordict):
        """Collect statistics over a trajectory of pointmaps to compute mean and std."""        
        for key, spec in self._input_specs.items():
            for name, stream in spec.streams.items():
                if not isinstance(stream, PointMapStream):
                    continue

                pointmap = tensordict["obs", key][name]
                
                if not stream.channel_order == "HWC":
                    pointmap = torch.movedim(pointmap, -1, -3)

                # (..., H, W, C) -> (N, C)
                pointmap = pointmap.flatten(0, -2)
                assert pointmap.dim() == 2

                # We use Welford's algorithm to compute mean and variance online
                # we do it in a batched manner for efficiency
                n_points = pointmap.shape[0]
                self.n_points += n_points
                
                delta = pointmap - self.mean
                self.mean += delta.sum(dim=0) / self.n_points
                delta2 = pointmap - self.mean
                self.m2 += torch.sum(delta * delta2, dim=0)
                print(self.m2)

        return tensordict
    
    def forward(self, tensordict):
        """Normalize pointmaps using the computed mean and std."""
        if self.n_points < 2:
            raise RuntimeError("Not enough points to compute statistics. Call `call_trajectory` first.")
        
        mean = self.mean
        var = self.m2 / (self.n_points - 1)
        std = torch.sqrt(var)

        for key, spec in self._input_specs.items():
            pointmaps = tensordict["obs", key]
            for name, stream in spec.streams.items():
                if not isinstance(stream, PointMapStream):
                    continue

                pointmap = pointmaps[name]
                if stream.channel_order == "HWC":
                    pointmap = torch.movedim(pointmap, -1, -3)

                pointmap = F.normalize(pointmap, mean, std, inplace=True)
                
                if stream.channel_order == "HWC":
                    pointmap = torch.movedim(pointmap, -3, -1)
                    
                pointmaps[name] = pointmap

        return tensordict
    
    def reverse(self, tensordict):
        """Do nothing on reverse."""
        return tensordict

    
    
if __name__ == "__main__":
    # Let's create a dummy specs and test the Normalize transform
    spec = DataSpecs(
        obs={
            "camera1": CameraSpec(
                streams={
                    "pointmap": PointMapStream(height=64, width=64, channels=3, channel_order="HWC")
                }
            )
        },
        action={},
    )
    
    transform = PointMapNormalize(spec)
    t1 = torch.randn(3, 10, 64, 64, 3) * 3 + 2 # A batch of 10 pointmaps
    t2 = torch.randn(3, 15, 64, 64, 3) * 3 + 2 # A batch of 15 pointmaps
    
    td1 = TensorDict({"obs": {"camera1": {"pointmap": t1}}})
    td2 = TensorDict({"obs": {"camera1": {"pointmap": t2}}})
        
    transform.call_trajectory(td1)
    transform.call_trajectory(td2)
        
    print(torch.mean(transform(td1)["obs", "camera1"]["pointmap"], dim=(0,1,2,3)))
