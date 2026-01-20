from __future__ import annotations

import logging
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from agents.beso_agent import BesoAgent as BaseBesoAgent
from agents.edm_diffusion.gc_sampling import NoiseScheduleType, SamplerType
from agents.edm_diffusion.noise_distributions import NoiseDistributionType
from environments.specs import DataSpecs
from transforms.base_transform import (
    Compose,
    Sequential,
    TransformPartial,
    TransformPartialsDict,
)
from utils.nested import unflatten_nested_tensor
from utils.tensors import unsqueeze_to

log = logging.getLogger(__name__)


class BesoAgent(BaseBesoAgent):

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
        precondition_type: str = "edm",
    ):
        super().__init__(
            noise_model=noise_model,
            noise_distribution=noise_distribution,
            noise_schedule=noise_schedule,
            sampler=sampler,
            obs_encoder=obs_encoder,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            specs=specs,
            num_sampling_steps=num_sampling_steps,
            sigma_data=sigma_data,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            ema_decay=ema_decay,
            goal_encoder=goal_encoder,
            normalizer=normalizer,
            reverse_transform=reverse_transform,
            precondition_type=precondition_type,
        )
        self.num_timesteps = specs.action_seq_len

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
            error = F.mse_loss(prediction, batch["ref_action"])
            self.log("val_action_mse", error, batch_size=batch["obs"].shape[0])

        # return the prediction in case we want to write it back to the environment
        return prediction
