from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from tensordict import TensorDict
    from torch.nn import Module

    from environments.specs import DataSpecs

log = logging.getLogger(__name__)


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        depth: Callable[[DataSpecs], Module] | None,
        pixels: Callable[[DataSpecs], Module] | None,
        tokenizer: Callable[[DataSpecs], Module],
        embed_dim: int,
        robot_state_encoder: Callable[[int, int], Module] | None = None,
    ):
        super().__init__()

        # BALAZS: propagate specs through sequence
        self.depth = depth(specs) if depth is not None else lambda x: x
        self.pixels = pixels(specs) if pixels is not None else lambda x: x
        self.tokenizer = tokenizer(specs)

        specs = self.tokenizer.specs
        obs_specs = specs.obs

        # if we encode the states, add additional tokens for each observed time step
        if robot_state_encoder is not None:
            robot_state_shape = specs.obs["robot_state"].shape
            state_seq_len, state_dim = robot_state_shape
            robot_state_encoder = robot_state_encoder(state_dim, embed_dim)

            # increase length of embedding sequence to account for state tokens
            obs_specs = dict(obs_specs)  # copy obs specs for local modification
            embed_spec = obs_specs["obs_embed"]
            obs_specs["obs_embed"] = dataclasses.replace(
                embed_spec,
                shape=(embed_spec.shape[0] + state_seq_len,) + embed_spec.shape[1:],
            )

        self.robot_state_encoder = robot_state_encoder
        self._specs = dataclasses.replace(specs, obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def forward(self, batch: TensorDict) -> TensorDict:
        batch = self.depth(batch)
        batch = self.pixels(batch)

        # batch["obs_embed"]: [B,T,N,D]
        batch = self.tokenizer(batch)

        # maybe compute robot state embeddings
        if self.robot_state_encoder is not None:
            robot_state = batch["obs", "robot_state"]
            leading_dims, state_dim = robot_state.shape[:-1], robot_state.shape[-1]
            # [B,T,M] -> [B*T,M]
            robot_state = robot_state.view(-1, state_dim)

            # [B*T,M] -> [B*T,D]
            state_emb = self.robot_state_encoder(robot_state)

            # [B*T,D] -> [B,T,1,D]
            state_emb = state_emb.view(*leading_dims, 1, -1)
            # concatenate along N dimension of embedding, keeping tokens from the same time step together
            embedding = batch["obs", "obs_embed"]
            batch["obs", "obs_embed"] = torch.cat([embedding, state_emb], dim=2)

        return batch
