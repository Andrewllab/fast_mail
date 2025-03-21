from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

from utils.logging import warn_once

if TYPE_CHECKING:
    from torch import Tensor

    from environments.dataset.base_dataset import TrajectoryDataset

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

        self.camera_names = dataset.camera_names

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, obs: dict) -> Tensor:
        # flatten batch and time dimensions of all camera images
        for camera in self.camera_names:
            # BALAZS: replace this with einops

            # should have shape [B, T, C, H, W]
            assert obs[f"{camera}_rgb"].ndim == 5
            # BALAZS: when is T ever not 1?
            assert obs[f"{camera}_rgb"].shape[1] == 1

            # should have shape [B, T, C, H, W]
            assert obs[f"{camera}_depth"].ndim == 5
            assert obs[f"{camera}_depth"].shape[2] == 1  # only one channel
            # BALAZS: when is T ever not 1?
            assert obs[f"{camera}_depth"].shape[1] == 1

            obs[f"{camera}_rgb"] = (
                obs[f"{camera}_rgb"]
                .permute(0, 3, 1, 2)
                .float()
                .div(255.0)
                .flatten(start_dim=0, end_dim=1)
            )

            obs[f"{camera}_depth"] = obs[f"{camera}_depth"].flatten(
                start_dim=0, end_dim=1
            )

        obs = self.depth(obs)
        obs = self.pixels(obs)
        embedding = self.tokenizer(obs)

        # maybe compute robot state embeddings
        if self.robot_state_encoder is not None:
            if "robot_state" in obs:
                state_emb = self.robot_state_encoder(obs["robot_state"])
                embedding = torch.cat([embedding, state_emb], dim=1)
            else:
                warn_once(
                    log,
                    "A robot state encoder has been instantiated, but the data does not contain a `robot_state` field!",
                )

        return embedding
