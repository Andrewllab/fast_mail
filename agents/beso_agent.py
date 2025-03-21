from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Iterable, Type

import torch
import torch.nn as nn

from agents.base_agent import BaseAgent
from agents.beso.edm_diffusion.utils import unsqueeze_to

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from agents.beso.edm_diffusion.gc_sampling import NoiseScheduleType, SamplerType
    from agents.beso.edm_diffusion.noise_distributions import NoiseDistributionType
    from agents.utils.scaler import Scaler
    from environments.dataset.base_dataset import TrajectoryDataset


log = logging.getLogger(__name__)


class BesoAgent(BaseAgent):
    def __init__(
        self,
        noise_model: Callable[[TrajectoryDataset], nn.Module],
        noise_distribution: NoiseDistributionType,
        noise_schedule: NoiseScheduleType,
        sampler: SamplerType,
        obs_encoder: Callable[[TrajectoryDataset], nn.Module],
        optimizer: Callable[[Iterable[Tensor]], Optimizer],
        lr_scheduler: Callable[[Optimizer], LRScheduler] | None,
        scaler: Type[Scaler],
        language_encoder: nn.Module | None,
        dataset: TrajectoryDataset,
        num_sampling_steps: int,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        ema_decay: float = 0.0,
    ):
        super().__init__(
            model=noise_model,
            obs_encoder=obs_encoder,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            scaler=scaler,
            language_encoder=language_encoder,
            dataset=dataset,
            ema_decay=ema_decay,
        )

        self.noise_distribution = noise_distribution
        self.noise_schedule = noise_schedule
        self.sampler = sampler

        self.num_sampling_steps = num_sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.action_dim = dataset.action_dim
        self.act_seq_len = dataset.act_seq_len

    def training_step(self, batch, batch_idx) -> Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        obs_dict, action, mask = batch

        obs = self.obs_encoder(obs_dict)
        action = self.scaler.normalize(action)
        goal = self.encode_language(obs_dict)

        sigma = self.noise_distribution(shape=(len(action),), device=self.device)
        noise = torch.randn_like(action)

        c_skip, c_out, c_in = [
            unsqueeze_to(c, action) for c in self.get_scalings(sigma)
        ]
        noised_input = action + noise * unsqueeze_to(sigma, action)
        model_output = self.model(obs, noised_input * c_in, goal, sigma)
        target = (action - c_skip * noised_input) / c_out
        loss = (model_output - target).pow(2).mean()

        self.log_dict({"loss": loss}, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        """Denoise the next sequence of actions"""
        obs_dict, action, mask = batch

        obs = self.obs_encoder(obs_dict)
        goal = self.encode_language(obs_dict)

        sigmas = self.noise_schedule(self.num_sampling_steps, device=self.device)

        x = (
            torch.randn(
                (obs.shape[0], self.act_seq_len, self.action_dim),
                device=self.device,
            )
            * self.sigma_max
        )

        action = self.sampler(
            model=self,
            state=obs,
            action=x,
            goal=goal,
            sigmas=sigmas,
            scaler=None,  # scalar only used for clipping actions
        )

        return action

    def validation_step(self, batch, batch_idx):
        metrics = self._eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        metrics = self._eval_step(batch, batch_idx)

    def _eval_step(self, batch, batch_idx):
        actions = self.predict_step(batch, batch_idx)
        return {}

    def get_scalings(self, sigma):
        """
        Compute the scalings for the denoising process.

        Args:
            sigma: The input sigma.
        Returns:
            The computed scalings for skip connections, output, and input.
        """
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def forward(
        self, obs: Tensor, action: Tensor, goal: Tensor, sigma: Tensor
    ) -> Tensor:
        """Predict the noise in action as part of sampling."""

        c_skip, c_out, c_in = [
            unsqueeze_to(c, action) for c in self.get_scalings(sigma)
        ]
        return self.model(obs, action * c_in, goal, sigma) * c_out + action * c_skip
