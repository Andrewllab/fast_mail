from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Iterable, Type

import torch
import torch.nn as nn

from agents.base_agent import BaseAgent

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
        model: nn.Module,
        optimizer: Callable[[Iterable[Tensor]], Optimizer],
        lr_scheduler: Callable[[Optimizer], LRScheduler] | None,
        obs_encoder: Callable[[TrajectoryDataset], nn.Module],
        robot_state_encoder: nn.Module | None,
        scaler: Type[Scaler],
        language_encoder: nn.Module | None,
        dataset: TrajectoryDataset,
        noise_distribution: NoiseDistributionType,
        noise_schedule: NoiseScheduleType,
        sampler: SamplerType,
        num_sampling_steps: int,
        sigma_data: float,
        sigma_min: float,
        sigma_max: float,
        ema_decay: float = 0.0,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            obs_encoder=obs_encoder,
            robot_state_encoder=robot_state_encoder,
            scaler=scaler,
            language_encoder=language_encoder,
            dataset=dataset,
            ema_decay=ema_decay,
        )

        self.noise_distribution = noise_distribution
        self.noise_schedule = noise_schedule
        self.sampler = sampler

        # diffusion stuff
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
        obs_dict, actions, mask = batch

        perceptual_emb, latent_goal = self.encode_obs(obs_dict)
        actions = self.scaler.normalize(actions)

        sigmas = self.noise_distribution(shape=(len(actions),), device=self.device)
        noise = torch.randn_like(actions)

        # BALAZS: does this belong inside the model?
        loss, _ = self.model.loss(perceptual_emb, actions, latent_goal, noise, sigmas)

        self.log_dict({"loss": loss}, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        """Denoise the next sequence of actions"""
        obs_dict, actions, mask = batch
        perceptual_emb, latent_goal = self.encode_obs(obs_dict)

        # if len(latent_goal.shape) < len(
        #         perceptual_emb['state_images'].shape if isinstance(perceptual_emb, dict) else perceptual_emb.shape):
        #     latent_goal = latent_goal.unsqueeze(1)  # .expand(-1, seq_len, -1)

        input_state = perceptual_emb
        sigmas = self.noise_schedule(self.num_sampling_steps, device=self.device)

        x = (
            torch.randn(
                (len(perceptual_emb), self.act_seq_len, self.action_dim),
                device=self.device,
            )
            * self.sigma_max
        )

        actions = self.sampler(
            model=self.model,
            state=input_state,
            action=x,
            goal=latent_goal,
            sigmas=sigmas,
            scaler=None,  # scalar only used for clipping actions
        )

        return actions

    def validation_step(self, batch, batch_idx):
        metrics = self._eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        metrics = self._eval_step(batch, batch_idx)

    def _eval_step(self, batch, batch_idx):
        actions = self.predict_step(batch, batch_idx)
        return {}
