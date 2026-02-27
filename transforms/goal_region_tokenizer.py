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
        spatial_encoder: Callable[[int], nn.Linear] | None = None,
        obs_key: str = "goal_pos",
    ):
        super().__init__()

        if obs_key not in specs.obs:
            raise KeyError(f"Observation spec at key {obs_key} not found in specs.")

        goal_pos = specs.obs[obs_key]
        match goal_pos.shape:
            case T, goal_dim if goal_dim == 3:
                pass
            case _:
                raise ValueError(
                    f"Goal region at key {obs_key} must be of shape [T, 3]"
                )

        if spatial_encoder is not None:
            spatial_encoder = spatial_encoder(goal_dim)
            goal_dim = spatial_encoder.out_features
        self.spatial_encoder = spatial_encoder

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

    def _call_one(self, goal_pos: Tensor, obs_embed: Tensor | None) -> Tensor:

        # goal_pos: (B, T, 3)

        if self.spatial_encoder is not None:
            # goal_pos -> (B, T, D)
            goal_pos = self.spatial_encoder(goal_pos)

        # (B, T, D) -> (B, T, D)
        goal_emb = self.model(goal_pos)

        if obs_embed is None:
            return goal_emb

        if obs_embed.is_nested:
            return cat_nested([obs_embed, goal_emb], dim=-2)

        # concatenate along N dimension of embedding, keeping tokens from the same time step together
        return torch.cat([obs_embed, goal_emb], dim=-2)
