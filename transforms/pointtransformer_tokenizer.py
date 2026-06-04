from __future__ import annotations

import logging
from typing import Callable, Sequence

import torch
import torch.nn as nn
from tensordict import NonTensorData
from torch import Tensor
from torch_geometric.data import Batch, Data

from environments.specs import DataSpecs, EmbedSpec, PointCloudSpec
from models.pointtransformer import PointTransformer
from transforms.base_transform import KeyMapping, Transform
from utils.nested import cat_nested

log = logging.getLogger(__name__)


class PointTransformerTokenizer(Transform, nn.Module):
    """Tokenizer based on the paper `Point Transformer`
    <https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Point_Transformer_ICCV_2021_paper.html>` .
    """

    def __init__(
        self,
        specs: DataSpecs,
        embed_dim: int,
        hidden_sizes: Sequence[int],
        k: int,
        spatial_encoder: Callable[[int], nn.Linear] | None = None,
        pcd_key: str = "pcd",
        token_pos_encoder: Callable[[int, int], nn.Module] | None = None,
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

        if spatial_encoder is not None:
            spatial_encoder = spatial_encoder(point_dim)
            point_dim = spatial_encoder.out_features
        self.spatial_encoder = spatial_encoder

        if self._input_spec.color:
            point_dim += 3

        self.model = PointTransformer(
            in_channels=point_dim,
            out_channels=embed_dim,
            dim_model=list(hidden_sizes),
            k=k,
        )

        # create encoder for token position
        if token_pos_encoder is None:
            log.warning("No token position encoder provided. Using nn.Embedding.")
            token_pos_encoder = nn.Embedding
        self.token_pos_encoder = token_pos_encoder(1, embed_dim)

        # create a modified specs object for the output
        new_spec = EmbedSpec(embed_dim=embed_dim, n_tokens=1)
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

        assert isinstance(data, Data)
        assert isinstance(data, Batch)

        pos, batch, ptr, color = data.pos, data.batch, data.ptr, data.x
        batch_size = getattr(data, "batch_size", None)
        assert pos is not None
        assert batch is not None
        assert ptr is not None

        # features: (B*N, 3)
        features = pos

        if self.spatial_encoder is not None:
            # features: (B*N, D)
            features = self.spatial_encoder(features)

        if color is not None:
            features = torch.cat([features, color], dim=-1)

        features = self.model(features, pos, batch, batch_size)

        if features.size(0) != batch_size:
            raise RuntimeError(
                f"Expected output features to have batch size {batch_size}, but got {features.size(0)}. {pos.shape=}"
            )

        # features -> (B, 1, D)
        features = features.unsqueeze(1)

        # add encoding of the token position to the token
        token_indices = torch.arange(1, dtype=torch.long, device=features.device)
        token_pos_embed = self.token_pos_encoder(token_indices)
        features += token_pos_embed

        if obs_embed is None:
            return features

        # concatenate along N dimension of embedding
        # obs_embed: (B, N, D)
        return cat_nested([obs_embed, features], dim=-2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
