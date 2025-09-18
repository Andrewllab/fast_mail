from __future__ import annotations
from typing import Sequence, Union

import torch
from tensordict import NonTensorData
from torch_geometric.data import Data
from torch_geometric.transforms import RandomJitter

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform


class VariableJitter(RandomJitter):
    def __init__(self, translate):
        self.max_translate = translate
        super().__init__(translate)

    def __call__(self, data):
        self.translate = torch.rand(1).item() * self.max_translate
        return super().__call__(data)


class JitterPointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        max_sigma: Union[float, int, Sequence[Union[float, int]]],
        pcd_keys: str | Sequence[str] = "pcd",
    ) -> None:

        self.max_sigma = max_sigma

        self.jitter = VariableJitter(max_sigma)

        self._specs = specs
        if isinstance(pcd_keys, str):
            pcd_keys = [pcd_keys]
        else:
            pcd_keys = list(pcd_keys)
        self._pcd_keys = pcd_keys

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
        data: Data = nt_data.data

        pos = data.pos
        assert pos is not None

        data = self.jitter(data)

        return data
