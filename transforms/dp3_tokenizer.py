from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
from tensordict import NonTensorData
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn.aggr import MaxAggregation

from environments.specs import DataSpecs, EmbedSpec, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform
from utils.nested import cat_nested

log = logging.getLogger(__name__)


class DiffusionPolicy3DTokenizer(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        embed_dim: int,
        mlp_1: Callable[[int], nn.Module],
        mlp_2: Callable[[int], nn.Module],
        pcd_key: str = "pcd",
    ):
        super().__init__()

        self._input_key = pcd_key
        try:
            self._input_spec = specs.obs[pcd_key]
        except KeyError:
            raise ValueError(
                f"Key {pcd_key} not found in specs. Available keys: {list(specs.obs.keys())}"
            )
        if not isinstance(self._input_spec, PointCloudSpec):
            raise ValueError(
                f"Key {pcd_key} is not a point cloud spec. Found {self._input_spec.type}"
            )

        self.point_dim = 6 if self._input_spec.color else 3
        self.mlp_1 = mlp_1(self.point_dim)
        self.aggr = MaxAggregation()
        self.mlp_2 = mlp_2(embed_dim)

        if self.mlp_1.out_features != self.mlp_2.in_features:
            raise ValueError(
                f"The last layer of mlp_1 (size {self.mlp_1.out_features}) must equal the first layer of mlp_2 (size {self.mlp_2.in_features})"
            )

        new_spec = EmbedSpec(embed_dim=embed_dim, n_tokens=1)

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        if "embed" in obs_specs:
            embed_spec = obs_specs["embed"]
            assert isinstance(embed_spec, EmbedSpec)
            new_spec = embed_spec.concat(new_spec)
        obs_specs["embed"] = new_spec
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=[("obs", self._input_key), ("obs", "embed")],
                out_keys=[("obs", "embed")],
            )
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(self, nt_data: NonTensorData, obs_embed: Tensor | None) -> Tensor:
        data: Data = nt_data.data  # unpack NonTensorData wrapper around pyg Data object

        assert isinstance(data, Batch)
        pos, batch, color = data.pos, data.batch, data.x
        assert pos is not None
        assert batch is not None

        # features: (B*N, 3)
        features = pos
        if color is not None:
            features = torch.cat([features, color], dim=-1)

        # features -> (B*N, D)
        features = self.mlp_1(features)

        # features -> (B, D)
        features = self.aggr(features, batch)

        # features -> (B, D)
        features = self.mlp_2(features)

        # features -> (B, 1, D)
        pcd_embed = features.unsqueeze(1)

        if obs_embed is None:
            return pcd_embed

        # concatenate along N dimension of embedding
        # obs_embed: (B, N, D)
        return cat_nested([obs_embed, features], dim=-2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mlp_1={self.mlp_1},(mlp_2={self.mlp_2})"
