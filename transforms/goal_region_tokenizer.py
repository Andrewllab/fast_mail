from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from environments.specs import DataSpecs, EmbedSpec
from transforms.base_transform import KeyMapping, Transform
from utils.nested import cat_nested

log = logging.getLogger(__name__)


class GoalRegionEncoder(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        model: Callable[[int, int], Module],
        embed_dim: int,
        spatial_encoder: nn.Linear | None = None,
        obs_key: str = "goal_region",
    ):
        super().__init__()

        goal_region_shape = specs.obs[obs_key].shape
        if len(goal_region_shape) != 2:
            raise ValueError(f"Goal region at key {obs_key} must be of shape [T, goal_dim]")

        T, goal_dim = goal_region_shape

        assert goal_dim == 3, f"Goal region at key {obs_key} must have dimension 3, got {goal_dim}"

        self.spatial_encoder = spatial_encoder
        if self.spatial_encoder is not None:
            goal_dim = self.spatial_encoder.out_features

        # instantiate the model
        self.model = model(goal_dim, embed_dim)

        # each time step produces a single token
        new_spec = EmbedSpec(embed_dim=embed_dim, n_tokens=T)

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        if "embed" in obs_specs:
            # if embedding sequence has fixed length, increase length to account for state tokens
            embed_spec = obs_specs["embed"]
            assert isinstance(embed_spec, EmbedSpec)
            new_spec = embed_spec.concat(new_spec)
        obs_specs["embed"] = new_spec
        self._output_specs = specs.replace(obs=obs_specs)

        self._obs_key = obs_key

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=[("obs", self._obs_key), ("obs", "embed")],
                out_keys=[("obs", "embed")],
            )
        ]

    def _call_one(self, goal_region_state: Tensor, obs_embed: Tensor | None) -> Tensor:

        features = goal_region_state

        if self.spatial_encoder is not None:
            # features: (B*N, D)
            features = self.spatial_encoder(features)

        # (B, T, M) -> (B, N, D)
        goal_emb = self.model(features)

        if obs_embed is None:
            return goal_emb

        if obs_embed.is_nested:
            return cat_nested([obs_embed, goal_emb], dim=-2)

        # concatenate along N dimension of embedding, keeping tokens from the same time step together
        return torch.cat([obs_embed, goal_emb], dim=-2)
