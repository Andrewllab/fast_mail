from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Iterable, Type

import torch
import torch.nn.functional as F

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
    from environments.specs import DataSpecs
    from transforms.base_transform import TransformPartial, TransformPartialsDict


log = logging.getLogger(__name__)


class BesoAgent(BaseAgent):
    def __init__(
        self,
        noise_model: Callable[[DataSpecs], Module],
        noise_distribution: NoiseDistributionType,
        noise_schedule: NoiseScheduleType,
        sampler: SamplerType,
        obs_encoder: TransformPartialsDict,
        optimizer: Callable[[Iterable[Tensor]], Optimizer],
        lr_scheduler: Callable[[Optimizer], LRScheduler] | None,
        scaler: Type[Scaler],
        goal_encoder: TransformPartial | None,
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

        self.action_shape = self.specs.action.shape

    def training_step(self, batch, batch_idx) -> Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        batch = self.goal_encoder(batch)
        batch = self.obs_encoder(batch)

        obs, goal = batch["obs", "embed"], batch.get(("goal", "embed"), None)
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
        loss = F.mse_loss(model_output, target)

        # log these values per step and per epoch
        self.log_dict(
            {"loss": loss}, on_epoch=True, prog_bar=True, batch_size=batch.shape[0]
        )

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        """Denoise the next sequence of actions"""
        batch = self.goal_encoder(batch)
        batch = self.obs_encoder(batch)

        obs, goal = batch["obs", "embed"], batch.get(("goal", "embed"), None)

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

        action = self.scaler.unnormalize(action)

        return action

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        action = self.predict_step(batch, batch_idx)

        metrics = {}

        if "success" in batch:
            metrics["val_success"] = batch["success"]

        if "action" in batch:
            # only if we are validating on demonstration data
            error = F.mse_loss(action, batch["action"])
            metrics["val_action_mse"] = error

        # log these values per epoch
        self.log_dict(metrics, batch_size=batch.shape[0])

        # return the actions in case we want to write them back to the environment
        return action

    # validation and testing are identical
    test_step = validation_step

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
        if self.ema_decay > 0:
            from torch.optim.swa_utils import AveragedModel

            assert isinstance(self.model, AveragedModel)

        return self.model(obs, action * c_in, goal, sigma) * c_out + action * c_skip
