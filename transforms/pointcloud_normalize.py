from __future__ import annotations

import torch
from torch_geometric.data import Data

from environments.specs import DataSpecs, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform


class NormalizePointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        center: tuple[float, float, float] | None = None,
        scale: float | None = None,
    ) -> None:
        self.center = center
        self.scale = scale

        if center is None or scale is None:
            raise NotImplementedError("Both center and scale must be provided.")

        self._pcd_keys = [
            key for key, spec in specs.obs.items() if isinstance(spec, PointCloudSpec)
        ]
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

    def _call_one(self, nt_data: NonTensorData) -> Data:
        data: Data = nt_data.data  # unpack NonTensorData wrapper around pyg Data object

        assert data.pos is not None

        if self.center is None:
            # TODO: compute center for each point cloud in batch
            # center the point cloud at the origin
            center = data.pos[:, :3].mean(dim=-2, keepdim=True)
        else:
            center = torch.tensor(self.center, device=data.pos.device)
        data.pos[...] -= center

        if self.scale is None:
            # scale such that the maximum coordinate (in any dimension) is 1
            # this is equivalent to scaling the point cloud to fit in a unit cube
            max = (
                data.pos.abs()
                # can you believe that max does not take a tuple as a dim argument?
                .flatten(start_dim=-2)
                .max(dim=-1, keepdim=True)
                .values.unsqueeze(-1)
            )
            scale = 0.999999 / max
        data.pos[...] *= scale

        return data
