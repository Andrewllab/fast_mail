from __future__ import annotations

import itertools
import logging

import torch
from tensordict import TensorDict
from torch import Tensor
from torch_geometric.data import Batch, Data

from environments.specs import (
    CameraSpec,
    DataSpecs,
    PointCloudSpec,
    PointMapStream,
)
from transforms.base_transform import Transform
from transforms.to_pointcloud import flatten_and_collate

log = logging.getLogger(__name__)


class PointMapToPointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        out_key: str = "pcd",
    ):
        self._out_key = out_key

        pointmap_specs = {}
        for key, spec in specs.obs.items():
            if isinstance(spec, CameraSpec):
                pointmap_streams = {
                    name: stream
                    for name, stream in spec.streams.items()
                    if isinstance(stream, PointMapStream)
                }
                if pointmap_streams:
                    pointmap_specs[key] = (spec, pointmap_streams)

        if not pointmap_specs:
            raise ValueError("No pointmap streams found in observation specs.")

        self._input_specs = pointmap_specs

        # Determine if any pointmap has color
        has_color = any(
            stream.color
            for _, (_, streams) in pointmap_specs.items()
            for stream in streams.values()
        )
        self.color = has_color

        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        obs_specs[self._out_key] = PointCloudSpec(
            feature_dim=(6 if has_color else 3), 
            color=has_color
        )
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # collect pointmaps from all cameras
        all_pointmaps = []

        for key, (spec, pointmap_streams) in self._input_specs.items():
            for stream_name, stream in pointmap_streams.items():
                pointmap = tensordict["obs", key, stream_name]
                
                # pointmap shape: (..., H, W, 3) or (..., H, W, 6)
                all_pointmaps.append(pointmap)

        # stack all pointmaps
        # pointmaps: (..., N, H, W, 3/6)
        pointmaps_batch = torch.stack(all_pointmaps, dim=-4)

        # here we must enforce that the time dimension is a singleton, since
        # we don't know how to handle time sequences of point clouds
        if pointmaps_batch.ndim == 6 and pointmaps_batch.shape[1] != 1:
            raise ValueError(
                f"Point cloud has time dimension {pointmaps_batch.shape[1]} (shape: {pointmaps_batch.shape}). Time sequences of point clouds are not supported."
            )

        # flatten batch and time dimensions
        # pointmaps_batch: (..., N, H, W, 3/6) -> (B, N, H, W, 3/6)
        pointmaps_batch = torch.flatten(pointmaps_batch, end_dim=-5)
        
        rgb_batch = pointmaps_batch[..., 3:6] if self.color else None
        pointmaps_batch = pointmaps_batch[..., :3]
            
        
        batch = flatten_and_collate(pointmaps_batch, None, rgb_batch)

        tensordict["obs", self._out_key] = batch
        return tensordict