import logging
from turtle import pos
from typing import Sequence

import torch
from tensordict import NonTensorData, TensorDict
from torch_geometric.data import Batch, Data

from environments.specs import DataSpecs, NestedTensorSpec
from transforms.base_transform import KeyMapping, Transform
from utils.pyg import (
    apply_index,
    batch2ptr,
    fps,
    nested_tensor_to_pyg,
    update_batch_metadata,
)

log = logging.getLogger(__name__)


class FpsSamplePointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        ratio: float | None = None,
        n_points: int | None = None,
        random_start: bool = True,
        pcd_keys: str | Sequence[str] = "pcd",
    ) -> None:

        if ratio is not None and n_points is not None:
            raise ValueError("Only one of ratio or n_points can be set.")
        elif ratio is None and n_points is None:
            raise ValueError("One of ratio or n_points must be set.")
        self.ratio = ratio
        self.n_points = n_points
        self.random_start = random_start

        if isinstance(pcd_keys, str):
            pcd_keys = [pcd_keys]
        else:
            pcd_keys = list(pcd_keys)
        self._pcd_keys = pcd_keys

        self._specs = specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(in_keys=[("obs", key)], out_keys=[("obs", key)])
            for key in self._pcd_keys
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def _call_one(self, pcd: TensorDict) -> Data:
        points = pcd["points"]

        if points.ndim == 4:
            # Only use the first timestep if there are multiple timesteps
            points = torch.nested.nested_tensor(
                [x[:, 0] for x in points.unbind(0)], layout=torch.jagged
            )

        pyg_points, ptr, batch = nested_tensor_to_pyg(points, return_batch=True)

        if pyg_points.numel() == 0:
            return pcd

        # pos: (B*N, 3)
        # idxs: (B*C)
        idxs = fps(
            pyg_points,
            ptr=ptr,
            ratio=self.ratio,
            n_points=self.n_points,
            random_start=self.random_start,
        )

        # Ensure proper dtype for indexing (and on same device)
        if idxs.dtype != torch.long:
            idxs = idxs.long()

        # Calculate new jaggedness metadata based on fps results and batch tensor
        # lengths are now the number of points sampled from each pointcloud, which can be calculated from the batch tensor by counting how many times each batch index appears in the idxs

        # Build new offsets from sampled indices
        B = ptr.numel() - 1
        new_lengths = torch.bincount(batch.index_select(0, idxs), minlength=B)

        new_offsets = torch.empty(B + 1, device=ptr.device, dtype=ptr.dtype)
        new_offsets[0] = 0
        new_offsets[1:] = new_lengths.cumsum(0)

        # Gather all fields with one shared idxs/new_offsets
        new_td = TensorDict({}, device=pyg_points.device)

        for key, v in pcd.items():
            # if idxs.max() > v.values().shape[0]:
            #     print(
            #         key,
            #         v.values().shape,
            #         v.offsets().shape,
            #         v.offsets().max(),
            #         idxs.max(),
            #         flush=True,
            #     )

            gathered = v.values().index_select(0, idxs)
            new_td[key] = torch.nested.nested_tensor_from_jagged(
                gathered, offsets=new_offsets
            )

        return new_td

    def __repr__(self) -> str:
        if self.ratio is not None:
            args = f"ratio={self.ratio}"
        else:
            args = f"n_points={self.n_points}"
        return f"{self.__class__.__name__}({args})"
