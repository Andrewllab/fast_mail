import logging
from turtle import pos
from typing import Sequence

import torch
from tensordict import NonTensorData, TensorDict
from torch_geometric.data import Batch, Data

from environments.specs import DataSpecs, NestedTensorSpec
from transforms.base_transform import KeyMapping, Transform
from utils.nested import cat_nested
from utils.pyg import (
    apply_index,
    batch2ptr,
    fps,
    nested_tensor_to_pyg,
    update_batch_metadata,
)

log = logging.getLogger(__name__)


class NestedPointCloudMergerTransform(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        pointcloud_keys: str | Sequence[str] = "pcd",
        out_key: str | None = None,
        overwrite_old_keys_with_empty: bool = True,
    ) -> None:

        self.out_key = out_key
        self.pointcloud_keys = pointcloud_keys
        self.overwrite_old_keys_with_empty = overwrite_old_keys_with_empty
        if isinstance(pointcloud_keys, str):
            pointcloud_keys = [pointcloud_keys]

        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> Data:
        pcds = []
        for key in self.pointcloud_keys:
            pcds.append(tensordict["obs", key])

        B = pcds[0].batch_size[0]
        device = pcds[0]["points"].device

        # Make sure all pointclouds have the same keys and batch size
        batch_size = pcds[0].batch_size
        keys = set(pcds[0].keys())
        total_points = torch.diff(pcds[0]["points"].offsets())
        for pcd in pcds[1:]:
            assert (
                pcd.batch_size == batch_size
            ), "All pointclouds must have the same batch size."
            assert (
                set(pcd.keys()) == keys
            ), f"All pointclouds must have the same keys. Got {set(pcd.keys())} and {keys}"
            total_points += torch.diff(pcd["points"].offsets())

        res_pcd = TensorDict({}, batch_size=batch_size)
        for key in keys:
            res_pcd[key] = cat_nested(
                [pcd[key] for pcd in pcds], dim=pcds[0][key]._ragged_idx
            )

        if self.overwrite_old_keys_with_empty:
            for pcd_key in self.pointcloud_keys:
                for key in keys:
                    points_val = tensordict["obs", pcd_key, key]
                    empty_off = torch.zeros((B + 1,), device=device, dtype=torch.long)
                    empty_vals = torch.empty(
                        (0, tensordict["obs", pcd_key, key].shape[-1]),
                        device=device,
                        dtype=points_val.dtype,
                    )
                    tensordict["obs", pcd_key, key] = (
                        torch.nested.nested_tensor_from_jagged(
                            empty_vals, offsets=empty_off
                        )
                    )
                    pass

        if self.out_key is not None:
            tensordict["obs", self.out_key] = res_pcd

        return tensordict
