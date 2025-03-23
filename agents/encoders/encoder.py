from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor

    from environments.datasets.base_dataset import TrajectoryDataset

log = logging.getLogger(__name__)


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        depth: Callable[[TrajectoryDataset], nn.Module] | None,
        pixels: Callable[[TrajectoryDataset], nn.Module] | None,
        tokenizer: Callable[[TrajectoryDataset], nn.Module],
        dataset: TrajectoryDataset,
        embed_dim: int,
        robot_state_encoder: nn.Module | None = None,
    ):
        super().__init__()

        self.depth = depth(dataset) if depth is not None else lambda x: x
        self.pixels = pixels(dataset) if pixels is not None else lambda x: x
        self.tokenizer = tokenizer(dataset)
        self.robot_state_encoder = robot_state_encoder

        self._embed_dim = embed_dim

        self.obs_space = dataset.obs_space
        self.rgb_obs_space = [
            key for key, info in self.obs_space.items() if info["type"] == "rgb"
        ]

        self._embed_seq_len: int = self.tokenizer.embed_seq_len

        # if we encode the states, add additional tokens for each observed time step
        if self.robot_state_encoder is not None:
            try:
                self._embed_seq_len += self.obs_space["robot_state"].shape[0]
            except KeyError:
                log.error(
                    "A robot state encoder has been instantiated, but the data does not contain a `robot_state` field!"
                )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def embed_seq_len(self) -> int:
        """Number of tokens in the output embedding."""
        return self._embed_seq_len

    def forward(self, obs: dict) -> Tensor:
        # flatten batch and time dimensions of all camera images
        for camera in self.rgb_obs_space:
            # BALAZS: replace this with einops

            # should have shape [B, T, C, H, W]
            assert obs[f"{camera}_rgb"].ndim == 5
            assert obs[f"{camera}_rgb"].shape[1] == 1

            # should have shape [B, T, C, H, W]
            assert obs[f"{camera}_depth"].ndim == 5
            assert obs[f"{camera}_depth"].shape[2] == 1  # only one channel
            assert obs[f"{camera}_depth"].shape[1] == 1

            # BALAZS: move to dataset transform (on_after_batch_transfer)
            obs[f"{camera}_rgb"] = (
                obs[f"{camera}_rgb"].permute(0, 3, 1, 2).float().div(255.0)
            )

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
