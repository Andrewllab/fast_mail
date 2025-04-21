from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
from tensordict import NonTensorData
from torch import Tensor
from torch_geometric.data import Batch, Data

from environments.specs import DataSpecs, EmbedSpec, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform
from utils.nested import cat_nested, pyg_to_nested_tensor

log = logging.getLogger(__name__)


class PointPatchTokenizer(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        embed_dim: int,
        mlp_1: Callable[[int], nn.Linear],
        mlp_2: Callable[[int], nn.Linear],
        position_encoder: Callable[[int, int], nn.Linear],
        pos_projection: nn.Linear | None = None,
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

        point_dim = 3

        self.pos_projection = pos_projection
        if self.pos_projection is not None:
            point_dim = self.pos_projection.out_features

        self.pos_encoder = position_encoder(point_dim, embed_dim)

        if self._input_spec.color:
            point_dim += 3

        self.mlp_1 = mlp_1(point_dim)
        self.mlp_2 = mlp_2(embed_dim)

        if self.mlp_1.out_features * 2 != self.mlp_2.in_features:
            raise ValueError(
                f"The last layer of mlp_1 (size {self.mlp_1.out_features}) must be half the size of the first layer of mlp_2 (size {self.mlp_2.in_features})"
            )

        new_spec = EmbedSpec(embed_dim=embed_dim, fixed_shape=False)

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
        center_pos, center_batch = data.pos, data.batch
        patch_pos, patch_color = data.patch_pos, data.get("patch_color")

        # features: (B*C, G, 3)
        features = patch_pos

        if self.pos_projection is not None:
            # features -> (B*C, G, D)
            features = self.pos_projection(features)
            # center_pos -> (B*C, D)
            center_pos = self.pos_projection(center_pos)

        if patch_color is not None:
            # concatenate color as an additional feature to the position
            patch_color = patch_color.to(dtype=features.dtype)
            features = torch.cat([features, patch_color], dim=-1)

        features = self.mlp_1(features)  # features -> (B*C, G, D)

        # max pool over each patch
        aggr_features = torch.max(features, dim=1, keepdim=True).values

        # add the neighborhood max to the original features for each node
        aggr_features = aggr_features.expand(-1, features.shape[-2], -1)
        # features -> (B*C, G, 2*D)
        features = torch.cat([aggr_features, features], dim=-1)

        features = self.mlp_2(features)  # features -> (B*C, G, D)

        # max pool over each patch
        features = torch.max(features, dim=1).values  # features -> (B*C, D)

        # add positional encoding
        features += self.pos_encoder(center_pos)

        pcd_embed = pyg_to_nested_tensor(features, batch=center_batch)

        if obs_embed is None:
            return pcd_embed

        # concatenate along N dimension of embedding
        # obs_embed: (B, N, D)
        return cat_nested([obs_embed, features], dim=-2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mlp_1={self.mlp_1},(mlp_2={self.mlp_2})"
