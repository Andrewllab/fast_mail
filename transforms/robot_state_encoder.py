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
        state_seq_len, state_dim = robot_state_shape
        self.model = model(state_dim, embed_dim)

        # create a modified specs object for the output
        # increase length of embedding sequence to account for state tokens
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        embed_spec = obs_specs["embed"]
        obs_specs["embed"] = dataclasses.replace(
            embed_spec,
            shape=(embed_spec.shape[0] + state_seq_len,) + embed_spec.shape[1:],
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

    def _call_one(self, robot_state: Tensor, obs_embed: Tensor) -> Tensor:
        # (B, T, M) -> (B, T, D)
        state_emb = self.model(robot_state)

        # (B, T, D) -> (B, T, N, D)
        state_emb = state_emb.unsqueeze(-2)
        # concatenate along N dimension of embedding, keeping tokens from the same time step together
        obs_embed = torch.cat([obs_embed, state_emb], dim=-2)
        return obs_embed
