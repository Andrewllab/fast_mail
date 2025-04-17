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


class RobotStateEncoder(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        model: Callable[[int, int], Module],
        embed_dim: int,
        obs_key: str = "robot_state",
    ):
        super().__init__()

        robot_state_shape = specs.obs[obs_key].shape
        if len(robot_state_shape) != 2:
            raise ValueError(f"Robot state at key {obs_key} must be of shape [T,M]")

        T, M = robot_state_shape

        # instantiate the model
        self.model = model(M, embed_dim)

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

    def _call_one(self, robot_state: Tensor, obs_embed: Tensor | None) -> Tensor:
        # (B, T, M) -> (B, N, D)
        state_emb = self.model(robot_state)

        if obs_embed is None:
            return state_emb

        if obs_embed.is_nested:
            return cat_nested([obs_embed, state_emb], dim=-2)

        # concatenate along N dimension of embedding, keeping tokens from the same time step together
        return torch.cat([obs_embed, state_emb], dim=-2)
