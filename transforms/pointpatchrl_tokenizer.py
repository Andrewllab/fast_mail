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
        token_pos_encoder: Callable[[int, int], nn.Linear],
        spatial_encoder: Callable[[int], nn.Linear] | None = None,
        input_key: str = "pcd",
        output_key: str = "embed",
    ):
        super().__init__()

        try:
            self._input_spec = specs.obs[input_key]
        except KeyError:
            raise ValueError(
                f"Key {input_key} not found in specs. Available keys: {list(specs.obs.keys())}"
            )
        if not isinstance(self._input_spec, PointCloudSpec):
            raise ValueError(
                f"Key {input_key} is not a point cloud spec. Found {self._input_spec.type}"
            )
        self.input_key = input_key
        self.output_key = output_key

        point_dim = 3

        if spatial_encoder is not None:
            spatial_encoder = spatial_encoder(point_dim)
            point_dim = spatial_encoder.out_features
        self.spatial_encoder = spatial_encoder

        self.token_pos_encoder = token_pos_encoder(point_dim, embed_dim)

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
                in_keys=[("obs", self.input_key), ("obs", self.output_key)],
                out_keys=[("obs", self.output_key)],
            )
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(self, nt_data: NonTensorData, obs_embed: Tensor | None) -> Tensor:
        data: Data = nt_data.data  # unpack NonTensorData wrapper around pyg Data object

        assert isinstance(data, Data)
        assert isinstance(data, Batch)
        center_pos, center_batch = data.pos, data.batch
        assert center_pos is not None
        assert center_batch is not None
        point_pos, patch_color = data.relative_pos, data.x

        # features: (B*C, G, 3)
        features = point_pos

        if self.spatial_encoder is not None:
            # features -> (B*C, G, D)
            features = self.spatial_encoder(features)
            # center_pos -> (B*C, D)
            center_pos = self.spatial_encoder(center_pos)

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

        # add encoding of the center position of the token to the token
        center_pos = self.token_pos_encoder(center_pos)
        features += center_pos

        pcd_embed = pyg_to_nested_tensor(features, batch=center_batch)

        if obs_embed is None:
            return pcd_embed

        # concatenate along N dimension of embedding
        # obs_embed: (B, N, D)
        return cat_nested([obs_embed, pcd_embed], dim=-2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mlp_1={self.mlp_1},(mlp_2={self.mlp_2})"

    @property
    def point_encoders(self) -> dict[str, Callable[[Tensor], Tensor]]:

        def mlp1(points: Tensor) -> Tensor:
            """Encode individual points using mlp_1 (and spatial encoder if
            present), exactly as in forward pass.
            """
            if self.spatial_encoder is not None:
                points = self.spatial_encoder(points)
            points = self.mlp_1(points)
            return points

        def patch_encoder(points: Tensor) -> Tensor:
            """Encode individual points using mlp_1 and mlp_2 (and spatial encoder
            if present), as if the patch only contained a single point. Max
            pooling is skipped.
            """
            features = points
            if self.spatial_encoder is not None:
                features = self.spatial_encoder(features)
            features = self.mlp_1(features)
            # for a single point, we skip the max pooling step and just
            # duplicate the features
            features = torch.cat([features, features], dim=-1)
            features = self.mlp_2(features)
            return features

        def patch_pos_encoder(points: Tensor) -> Tensor:
            """Encode individual points as if they were patch centers, using
            the token position encoder (and spatial encoder if present)."""
            if self.spatial_encoder is not None:
                points = self.spatial_encoder(points)
            return self.token_pos_encoder(points)

        callables = {
            "mlp1": mlp1,
            "patch_encoder": patch_encoder,
            "patch_pos_encoder": patch_pos_encoder,
        }

        if self.spatial_encoder is not None:
            callables["spatial_encoder"] = self.spatial_encoder

        return callables
