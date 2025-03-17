from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


class ObservationEncoder(nn.Module):
    def __init__(
        self,
        depth: nn.Module,
        pixels: nn.Module,
        tokenizer: nn.Module,
        camera_names: Sequence[str],
    ):
        super().__init__()

        self.depth = depth
        self.pixels = pixels
        self.tokenizer = tokenizer

        self.camera_names = camera_names

    def forward(self, obs: dict, latent_goal: Any) -> tuple[Tensor, Any]:
        # BALAZS: should rgb and depth be separate keys so they can have separate resolutions and data types?

        # flatten batch and time dimensions of all camera images
        for camera in self.camera_names:
            # should have shape [B, T, C, H, W]
            assert obs[f"{camera}_rgb"].ndim == 5
            # BALAZS: when is T ever not 1?
            assert obs[f"{camera}_rgb"].shape[1] == 1

            # should have shape [B, T, C, H, W]
            assert obs[f"{camera}_depth"].ndim == 5
            assert obs[f"{camera}_depth"].shape[2] == 1  # only one channel
            # BALAZS: when is T ever not 1?
            assert obs[f"{camera}_depth"].shape[1] == 1

            obs[f"{camera}_rgb"] = obs[f"{camera}_rgb"].flatten(start_dim=0, end_dim=1)
            obs[f"{camera}_depth"] = obs[f"{camera}_depth"].flatten(
                start_dim=0, end_dim=1
            )

        obs = self.depth(obs)
        obs = self.pixels(obs)
        obs = self.tokenizer(obs)

        return obs
