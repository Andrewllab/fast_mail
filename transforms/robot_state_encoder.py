from __future__ import annotations

import dataclasses
import logging
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from environments.specs import DataSpecs, EmbedSpec
from transforms.base_transform import KeyMapping, Transform

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

        # the length of the embedding is the number of leading
        assert len(robot_state_shape) == 2
        T, state_dim = robot_state_shape
        n_embed_tokens = 1  # each robot state is a single token

        # instantiate the model
        self.model = model(state_dim, embed_dim)

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        if "embed" in obs_specs:
            # if embedding sequence has fixed length, increase length to account for state tokens
            embed_spec = obs_specs["embed"]
            assert isinstance(embed_spec, EmbedSpec)
            assert len(embed_spec.shape) == 3
            assert embed_spec.shape[0] == T

            if embed_spec.fixed_shape:
                assert embed_spec.shape[1] is not None
                obs_specs["embed"] = dataclasses.replace(
                    embed_spec,
                    shape=(
                        T,
                        embed_spec.shape[1] + n_embed_tokens,
                        embed_spec.shape[2],
                    ),
                )
                log.debug(
                    f"Extended obs embedding spec to {n_embed_tokens} tokens per time step",
                )
        else:
            # create new embedding spec
            obs_specs["embed"] = EmbedSpec(shape=(T, n_embed_tokens, embed_dim))
            log.debug(
                f"Created obs embedding spec with {n_embed_tokens} tokens per time step",
            )
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
        # (B, T, M) -> (B, T, D)
        state_emb = self.model(robot_state)

        # (B, T, D) -> (B, T, N, D)
        state_emb = state_emb.unsqueeze(-2)

        if obs_embed is not None:
            # concatenate along N dimension of embedding, keeping tokens from the same time step together
            obs_embed = torch.cat([obs_embed, state_emb], dim=-2)
            return obs_embed
        else:
            return state_emb
