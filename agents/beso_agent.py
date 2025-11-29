from __future__ import annotations

import logging
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from agents.base_agent import BaseAgent
from agents.edm_diffusion.gc_sampling import NoiseScheduleType, SamplerType
from agents.edm_diffusion.noise_distributions import NoiseDistributionType
from environments.specs import DataSpecs
from transforms.base_transform import (
    Compose,
    Sequential,
    TransformPartial,
    TransformPartialsDict,
)
from utils.tensors import unsqueeze_to

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
        specs: DataSpecs,
        num_sampling_steps: int,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        ema_decay: float = 0.0,
        goal_encoder: TransformPartial | None = None,
        normalizer: Sequential | None = None,
        reverse_transform: Compose | None = None,
    ):
        super().__init__(
            model=noise_model,
            obs_encoder=obs_encoder,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            specs=specs,
            ema_decay=ema_decay,
            goal_encoder=goal_encoder,
            normalizer=normalizer,
            reverse_transform=reverse_transform,
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
        batch = self.normalizer(batch)
        batch = self.goal_encoder(batch)
        batch = self.obs_encoder(batch)

        obs, goal = batch["obs", "embed"], batch.get(("goal", "embed"), None)
        action = batch["action"]

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
        batch = self.normalizer(batch)
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
            model=self,  # call self.forward to evaluate the model
            state=obs,
            action=x,
            goal=goal,
            sigmas=sigmas,
            scaler=None,  # scalar only used for clipping actions
        )

        batch["action"] = action

        batch = self.reverser.reverse(batch)

        return batch["action"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # values logged here get averaged over an epoch

        if "episode_info" in batch:
            episode_info = batch["episode_info"]
            assert self.checkpoint_metadata and "epoch" in self.checkpoint_metadata
            self.log_dict(
                {
                    "ckpt_epoch": self.checkpoint_metadata["epoch"],
                    **episode_info.to_dict(),
                },
                batch_size=episode_info.shape[0],
            )

        prediction = self.predict_step(batch, batch_idx)

        if "ref_action" in batch:
            # only if we are validating on demonstration data
            error = F.mse_loss(prediction, batch["ref_action"])
            self.log("val_action_mse", error, batch_size=batch["obs"].shape[0])

        # return the prediction in case we want to write it back to the environment
        return prediction

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
