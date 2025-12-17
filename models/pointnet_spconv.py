from __future__ import annotations

from functools import partial
from typing import Callable, Sequence

import spconv.pytorch as spconv
import torch
import torch.nn as nn
from torch import Tensor

from utils.pyg import offset2ptr, ptr2batch


class PointNetSpconv(nn.Module):
    """
    PointNet with spconv.pytorch.

    Reference: https://github.com/HaoyiZhu/PointCloudMatters/blob/main/src/models/components/pcd_encoder/pointnet.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        channel_sizes: Sequence[int] | None = None,
        norm: Callable[[int], nn.Module] | str = "BatchNorm1d",
        activation: Callable[[], nn.Module] | str = "ReLU",
    ):
        super().__init__()
        self.in_channels = in_channels

        if channel_sizes is None:
            channel_sizes = [64, 64, 64, 128, 512]  # default from PCM pointnet

        if isinstance(activation, str):
            activation = getattr(nn, activation)
        if isinstance(norm, str):
            norm = getattr(nn, norm)

        norm_fn = partial(norm, eps=1e-3, momentum=0.01)

        layers = []
        current_channels = in_channels
        for size in channel_sizes:
            layers.append(
                spconv.SparseSequential(
                    spconv.SubMConv3d(
                        current_channels, size, kernel_size=1, bias=False
                    ),
                    norm_fn(size),
                    activation(),
                )
            )
            current_channels = size
        self.backbone = nn.Sequential(*layers)

        # final output layer
        if out_channels is not None and out_channels > 0:
            self.final = spconv.SubMConv3d(
                current_channels, out_channels, kernel_size=1, padding=1, bias=True
            )
            self._num_channels = out_channels
        else:
            self.final = spconv.Identity()
            self._num_channels = current_channels

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def forward(self, input_dict: dict[str, Tensor]) -> Tensor:

        feat = input_dict["feat"]
        grid_coord = input_dict["coord"]
        offset = input_dict["offset"]

        batch = ptr2batch(offset2ptr(offset))

        sparse_shape = torch.max(grid_coord, dim=0).values.int() + 96

        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape.tolist(),
            batch_size=len(offset),
        )

        x = self.backbone(x)
        x = self.final(x)

        return x.features


def pyg_to_spconv(data) -> dict[str, Tensor]:
    """
    Convert PyG data to spconv input format.
    """
    pos = data.pos  # (N, 3)
    batch = data.batch  # (N,)

    voxel_size = 0.05  # adjust as needed
    discrete_coords = torch.floor(pos / voxel_size).int()  # (N, 3)

    coord_min = discrete_coords.min(dim=0).values
    discrete_coords -= coord_min.unsqueeze(0)

    coord_with_batch = torch.cat(
        [batch.unsqueeze(-1), discrete_coords], dim=1
    )  # (N, 4)

    unique_coords, inverse_indices = torch.unique(
        coord_with_batch, return_inverse=True, dim=0
    )

    feat = data.x if data.x is not None else pos  # (N, C) or (N, 3)

    summed_feat = torch.zeros(
        (unique_coords.size(0), feat.size(1)), device=feat.device
    ).index_add_(0, inverse_indices, feat)

    counts = torch.zeros((unique_coords.size(0),), device=feat.device).index_add_(
        0, inverse_indices, torch.ones_like(inverse_indices, dtype=feat.dtype)
    )

    averaged_feat = summed_feat / counts.unsqueeze(-1)

    offset = torch.zeros(
        (batch.max().item() + 1,), device=feat.device, dtype=torch.long
    )
    for b in range(batch.max().item() + 1):
        offset[b] = (batch == b).sum()
    offset = torch.cumsum(offset, dim=0)

    return {
        "feat": averaged_feat,
        "coord": unique_coords[:, 1:],
        "offset": offset,
    }
