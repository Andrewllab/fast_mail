from __future__ import annotations

import logging

import torch.nn as nn
from tensordict import NonTensorData
from torch_geometric.data import Batch, Data

from environments.specs import DataSpecs, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform
from utils.pyg import batch2ptr, fps, knn, update_batch_metadata

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
        self.oversampling_ratio = oversampling_ratio
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

        pos, batch, ptr, features = data.pos, data.batch, data.ptr, data.x
        batch_size = getattr(data, "batch_size", None)
        assert pos is not None
        assert batch is not None
        assert ptr is not None

        if pos.numel() == 0:
            # empty point cloud

            # add empty fields to data with the correct shapes to match the
            # expected output
            patch_pos = pos.new_zeros(0, self.patch_size, pos.size(-1))

            if features is not None:
                features = features.unflatten(dim=0, sizes=(0, self.patch_size))

            data.x = features  # (B*C, G, D) or None
            data.patch_pos = patch_pos  # (B*C, G, 3)

            return data

        # verify that ptr is up to date
        if data.ptr[-1] != batch.shape[0]:
            ptr = batch2ptr(batch)

        # pos: (B*N, 3)
        # center_idxs: (B*C,)
        center_idxs = fps(
            pos,
            ptr=ptr,
            ratio=self.fps_sampling_ratio,
            random_start=self.fps_random_start,
        )

        center_pos = pos[center_idxs]  # center_pos: (B*C, 3)
        center_batch = batch[center_idxs]  # center_batch: (B*C,)
        n_patches = center_idxs.size(0)

        # find the nearest k points to each center point. these groups of k
        # points become the patches
        # patch_idxs: (B*C*G,)
        _, patch_idxs = knn(
            x=pos,
            y=center_pos,
            k=self.patch_size,  # G
            ptr_x=ptr,
            batch_y=center_batch,
            batch_size=batch_size,
            pad_too_small=True,
        )

        patch_pos = pos[patch_idxs]  # patch_pos: (B*C*G, 3)
        # patch_pos -> (B*C, G, 3)
        patch_pos = patch_pos.view(n_patches, self.patch_size, 3)

        if features is not None:
            features = features[patch_idxs]  # features -> (B*C*G, D)
            # features -> (B*C, G, D)
            features = features.view(n_patches, self.patch_size, -1)

        # package the point patches back into the Batch object, since we cannot
        # directly instantiate a new Batch object
        # TODO: index and reshape all fields of Data object
        data.pos = center_pos  # (B*C, 3)
        data.batch = center_batch  # (B*C,)
        data.x = features  # (B*C, G, D) or None
        data.patch_pos = patch_pos  # (B*C, G, 3)

        data = update_batch_metadata(data)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(patch_size={self.patch_size},(oversampling_ratio={self.oversampling_ratio})"
