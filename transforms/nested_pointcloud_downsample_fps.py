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
            points = torch.nested.nested_tensor_from_jagged(
                points.values()[:, 0, :], points.offsets()
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


def _build_new_td(
    pcd: TensorDict, idxs: torch.Tensor, batch: torch.Tensor, ptr: torch.Tensor
) -> TensorDict:
    """Shared helper: gather all pcd fields by flat index, return new TensorDict with updated offsets."""
    B = ptr.numel() - 1
    new_lengths = torch.bincount(batch.index_select(0, idxs), minlength=B)
    new_offsets = torch.empty(B + 1, device=ptr.device, dtype=ptr.dtype)
    new_offsets[0] = 0
    new_offsets[1:] = new_lengths.cumsum(0)

    new_td = TensorDict({}, device=idxs.device)
    for key, v in pcd.items():
        gathered = v.values().index_select(0, idxs)
        new_td[key] = torch.nested.nested_tensor_from_jagged(
            gathered, offsets=new_offsets
        )
    return new_td


class HybridFpsSamplePointCloud(Transform):
    """
    Hybrid FPS: splits budget between spatial FPS (xyz) and semantic FPS (DINOv2 features).

    Spatial FPS ensures global coverage of the scene geometry.
    Semantic FPS in feature space over-samples rare semantic regions
    (e.g. door handles) that are underrepresented by point count alone.

    Args:
        semantic_ratio: fraction of n_points drawn by feature-space FPS.
                        e.g. 0.5 => half spatial, half semantic.
    """

    def __init__(
        self,
        specs: DataSpecs,
        n_points: int,
        semantic_ratio: float = 0.5,
        random_start: bool = True,
        pcd_keys: str | Sequence[str] = "pcd",
        feature_key: str = "features",
    ) -> None:
        self.n_points = n_points
        self.semantic_ratio = semantic_ratio
        self.random_start = random_start
        self.feature_key = feature_key
        self._pcd_keys = [pcd_keys] if isinstance(pcd_keys, str) else list(pcd_keys)
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

    def _call_one(self, pcd: TensorDict) -> TensorDict:
        points = pcd["points"]
        features = pcd[self.feature_key]

        if points.ndim == 4:
            points = torch.nested.nested_tensor_from_jagged(
                points.values()[:, 0, :], points.offsets()
            )

        pyg_points, ptr, batch = nested_tensor_to_pyg(points, return_batch=True)
        if pyg_points.numel() == 0:
            return pcd

        n_semantic = int(self.n_points * self.semantic_ratio)
        n_spatial = self.n_points - n_semantic
        N_total = pyg_points.shape[0]

        if n_spatial > 0:
            # --- Spatial branch: FPS on xyz ---
            spatial_idxs = fps(
                pyg_points,
                ptr=ptr,
                n_points=n_spatial,
                random_start=self.random_start,
            ).long()

            all_idxs = torch.arange(N_total, device=pyg_points.device)
            remaining_flat_idxs = all_idxs[~torch.isin(all_idxs, spatial_idxs)]
        else:
            spatial_idxs = torch.tensor([], dtype=torch.long, device=pyg_points.device)
            remaining_flat_idxs = torch.arange(N_total, device=pyg_points.device)

        if n_semantic > 0:
            # --- Semantic branch: FPS on features, but only among points not already chosen by spatial FPS ---
            B = ptr.numel() - 1
            pyg_features = features.values()  # (N_total, F)
            remaining_features = pyg_features[remaining_flat_idxs]
            remaining_batch = batch[remaining_flat_idxs]

            remaining_lengths = torch.bincount(remaining_batch, minlength=B)
            remaining_ptr = torch.zeros(B + 1, device=ptr.device, dtype=ptr.dtype)
            remaining_ptr[1:] = remaining_lengths.cumsum(0)

            semantic_local_idxs = fps(
                remaining_features,
                ptr=remaining_ptr,
                n_points=n_semantic,
                random_start=self.random_start,
            ).long()

            # Map local remaining indices back to global flat indices
            semantic_idxs = remaining_flat_idxs[semantic_local_idxs]
        else:
            semantic_idxs = torch.tensor([], dtype=torch.long, device=pyg_points.device)

        combined_idxs, _ = torch.sort(torch.cat([spatial_idxs, semantic_idxs]))
        return _build_new_td(pcd, combined_idxs, batch, ptr)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_points={self.n_points}, "
            f"semantic_ratio={self.semantic_ratio})"
        )
