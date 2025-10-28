from __future__ import annotations

import logging

import torch
import torch.nn as nn
from tensordict import NonTensorData
from torch_geometric.data import Batch, Data
from torch_geometric.nn import fps, knn

from environments.specs import DataSpecs, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform
from utils.pyg import update_batch_metadata

log = logging.getLogger(__name__)


class PointPatcher(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        patch_size: int,
        oversampling_ratio: float,
        fps_random_start: bool = True,
        in_key: str = "pcd",
    ):
        super().__init__()

        self._input_key = in_key
        try:
            self._input_spec = specs.obs[in_key]
        except KeyError:
            raise ValueError(
                f"Key {in_key} not found in specs. Available keys: {list(specs.obs.keys())}"
            )
        if not isinstance(self._input_spec, PointCloudSpec):
            raise ValueError(
                f"Key {in_key} is not a point cloud spec. Found {self._input_spec.type}"
            )

        self.patch_size = patch_size
        self.fps_random_start = fps_random_start

        # N_patches * patch_size = N_points * oversampling_ratio
        # fps_sampling_ratio = N_patches / N_points
        # -> fps_sampling_ratio = oversampling_ratio / patch_size
        self.fps_sampling_ratio = oversampling_ratio / patch_size

        self._output_specs = specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=[("obs", self._input_key)],
                out_keys=[("obs", self._input_key)],
            )
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(self, nt_data: NonTensorData) -> Data:
        data: Data = nt_data.data  # unpack NonTensorData wrapper around pyg Data object

        assert isinstance(data, Data)
        assert isinstance(data, Batch)

        pos, batch, color = data.pos, data.batch, data.x
        assert pos is not None
        assert batch is not None

        # pos: (B*N, 3)
        # center_idxs: (B*C)
        center_idxs = fps(
            pos,
            data.batch,
            ratio=self.fps_sampling_ratio,
            random_start=self.fps_random_start,
            batch_size=data.batch_size,
        )

        center_pos = pos[center_idxs]  # center_pos: (B*C, 3)
        center_batches = batch[center_idxs]  # center_batches: (B*C, 3)

        # find the nearest k points to each center point. these groups of k
        # points become the patches
        # patch_idxs: (B*C*G)
        _, patch_idxs = knn(
            x=pos,
            y=center_pos,
            k=self.patch_size,  # G
            batch_x=batch,
            batch_y=center_batches,
            batch_size=data.batch_size,
        )

        patch_pos = pos[patch_idxs]  # patch_pos: (B*C*G, 3)
        patch_pos = patch_pos.view(-1, self.patch_size, 3)  # patch_pos -> (B*C, G, 3)

        # convert to relative coordinates around patch center
        # features: (B*C, G, 3)
        patch_pos = patch_pos - center_pos.unsqueeze(1)

        if color is not None:
            color = color[patch_idxs]  # color -> (B*C*G, 3)
            color = color.view(-1, self.patch_size, 3)  # color -> (B*C, G, 3)

        # package the point patches back into the Batch object, since we cannot
        # directly instantiate a new Batch object
        data.pos = center_pos
        data.batch = center_batches
        data = update_batch_metadata(data)
        data.x = None  # remove, since the patches do not have a unique color
        data["patch_pos"] = patch_pos
        data["patch_color"] = color

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mlp_1={self.mlp_1},(mlp_2={self.mlp_2})"
