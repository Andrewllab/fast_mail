from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Iterable, Type

import torch

from agents.base_agent import BaseAgent
from agents.edm_diffusion.utils import unsqueeze_to

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer

    from agents.edm_diffusion.gc_sampling import NoiseScheduleType, SamplerType
    from agents.edm_diffusion.noise_distributions import NoiseDistributionType
    from agents.utils.scaler import Scaler
    from environments.datasets.base_dataset import TrajectoryDataset
    from environments.specs import DataSpecs


log = logging.getLogger(__name__)


class BesoAgent(BaseAgent):
    def __init__(
        self,
        noise_model: Callable[[DataSpecs], Module],
        noise_distribution: NoiseDistributionType,
        noise_schedule: NoiseScheduleType,
        sampler: SamplerType,
        obs_encoder: Callable[[DataSpecs], Module],
        optimizer: Callable[[Iterable[Tensor]], Optimizer],
        lr_scheduler: Callable[[Optimizer], LRScheduler] | None,
        scaler: Type[Scaler],
        goal_encoder: Callable[[DataSpecs], Module] | None,
        specs: DataSpecs,
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
            goal_encoder=goal_encoder,
            specs=specs,
            ema_decay=ema_decay,
        )

        self.noise_distribution = noise_distribution
        self.noise_schedule = noise_schedule
        self.sampler = sampler

        self.num_sampling_steps = num_sampling_steps
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.action_shape = specs.action.shape

    def training_step(self, batch, batch_idx) -> Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        batch = self.encode_goal(batch)
        batch = self.obs_encoder(batch)
        obs, goal = batch.get(("obs", "obs_embed")), batch.get("goal_embed", None)

        action = batch["action"]
        action = self.scaler.normalize(action)
        sigma = self.noise_distribution(shape=(len(action),), device=self.device)
        noise = torch.randn_like(action)

        c_skip, c_out, c_in = [
            unsqueeze_to(c, action) for c in self.get_scalings(sigma)
        ]
        noised_input = action + noise * unsqueeze_to(sigma, action)
        model_output = self.model(obs, noised_input * c_in, goal, sigma)
        target = (action - c_skip * noised_input) / c_out
        loss = (model_output - target).pow(2).mean()

        self.log_dict({"loss": loss}, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        """Denoise the next sequence of actions"""
        obs_dict, action, mask = batch

        obs = self.obs_encoder(obs_dict)
        goal = self.encode_goal(obs_dict)

        sigmas = self.noise_schedule(self.num_sampling_steps, device=self.device)

        B = obs.shape[0]
        x = (
            torch.randn(
                (B, *self.action_shape),
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
