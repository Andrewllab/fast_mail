from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
from torch import Tensor

from environments.datasets.base_dataset import TrajectoryDataset
from environments.specs import DataSpecs

log = logging.getLogger(__name__)


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        depth: Callable[[DataSpecs], nn.Module] | None,
        pixels: Callable[[DataSpecs], nn.Module] | None,
        tokenizer: Callable[[DataSpecs], nn.Module],
        embed_dim: int,
        robot_state_encoder: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()

        # BALAZS: propagate specs through sequence
        self.depth = depth(specs) if depth is not None else lambda x: x
        self.pixels = pixels(specs) if pixels is not None else lambda x: x
        self.tokenizer = tokenizer(specs)

        specs = self.tokenizer.specs
        obs_spec = specs.obs

        # if we encode the states, add additional tokens for each observed time step
        if robot_state_encoder is not None:
            robot_state_shape = specs.obs["robot_state"].shape
            state_seq_len, state_dim = robot_state_shape
            robot_state_encoder = robot_state_encoder(state_dim, embed_dim)

            # increase length of embedding sequence to account for state tokens
            obs_spec = dict(obs_spec)  # copy obs spec for local modification
            spec = obs_spec["obs_embed"]
            obs_spec["obs_embed"] = dataclasses.replace(
                spec, shape=(spec.shape[0] + state_seq_len,) + spec.shape[1:]
            )

        self.robot_state_encoder = robot_state_encoder
        self._obs_spec = obs_spec
        self._specs = DataSpecs(_obs=self._obs_spec, action=specs.action)

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def forward(self, obs: dict) -> Tensor:
        # flatten batch and time dimensions of all camera images
        # for camera in self.rgb_keys:
        #     # BALAZS: replace this with einops

        #     # should have shape [B, T, C, H, W]
        #     assert obs[camera].ndim == 5
        #     assert obs[camera].shape[1] == 1

        #     # BALAZS: move to dataset transform (on_after_batch_transfer)
        #     obs[camera] = obs[camera].permute(0, 1, 4, 2, 3).float().div(255.0)

        obs = self.depth(obs)
        obs = self.pixels(obs)

        # embedding: [B,T,N,D]
        embedding = self.tokenizer(obs)

        # maybe compute robot state embeddings
        if self.robot_state_encoder is not None:
            robot_state = obs["robot_state"]
            leading_dims, state_dim = robot_state.shape[:-1], robot_state.shape[-1]
            # [B,T,M] -> [B*T,M]
            robot_state = robot_state.view(-1, state_dim)

            # [B*T,M] -> [B*T,D]
            state_emb = self.robot_state_encoder(robot_state)

            # [B*T,D] -> [B,T,1,D]
            state_emb = state_emb.view(*leading_dims, 1, -1)
            # concatenate along N dimension of embedding, keeping tokens from the same time step together
            embedding = torch.cat([embedding, state_emb], dim=2)

        return embedding
