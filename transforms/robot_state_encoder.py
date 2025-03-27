from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

from transforms.base_transform import KeyMapping, Transform

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

    from environments.specs import DataSpecs


class RobotStateEncoder(nn.Module, Transform):
    def __init__(
        self,
        specs: DataSpecs,
        model: Callable[[int, int], Module],
        embed_dim: int,
    ):
        super().__init__()

        robot_state_shape = specs.obs["robot_state"].shape
        state_seq_len, state_dim = robot_state_shape
        self.model = model(state_dim, embed_dim)

        # increase length of embedding sequence to account for state tokens
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        embed_spec = obs_specs["embed"]
        obs_specs["embed"] = dataclasses.replace(
            embed_spec,
            shape=(embed_spec.shape[0] + state_seq_len,) + embed_spec.shape[1:],
        )

        self._specs = dataclasses.replace(specs, obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=[("obs", "robot_state"), ("obs", "embed")],
                out_keys=[("obs", "embed")],
            )
        ]

    def forward(self, robot_state: Tensor, obs_embed: Tensor) -> Tensor:
        leading_dims, state_dim = robot_state.shape[:-1], robot_state.shape[-1]
        # [B,T,M] -> [B*T,M]
        robot_state = robot_state.view(-1, state_dim)

        # [B*T,M] -> [B*T,D]
        state_emb = self.model(robot_state)

        # [B*T,D] -> [B,T,1,D]
        state_emb = state_emb.view(*leading_dims, 1, -1)
        # concatenate along N dimension of embedding, keeping tokens from the same time step together
        obs_embed = torch.cat([obs_embed, state_emb], dim=2)
        return obs_embed
