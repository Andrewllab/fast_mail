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

    def edm_preconditioning(
        self, sigma: Tensor, action: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute the EDM scaling factors depending on the noise level. These
        factors adjust the learning objective to interpolate between predicting
        the denoised actions when sigma is high and predicting the noise when
        sigma is low. This ensures that the objective is roughly uniformly
        difficult across noise levels.

        See Table 1 in https://arxiv.org/pdf/2206.00364 for details.
        """
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5

        c_skip, c_out, c_in = [unsqueeze_to(c, action) for c in (c_skip, c_out, c_in)]

        c_noise = sigma.log() / 4
        return c_skip, c_out, c_in, c_noise

    def forward(self, batch: TensorDict, action: Tensor, sigma: Tensor) -> Tensor:
        """Predict the unnoised actions with the help of EDM preconditioning."""
        if self.ema_decay > 0 and not self.training:
            from torch.optim.swa_utils import AveragedModel

            assert isinstance(self.model, AveragedModel)

        c_skip, c_out, c_in, c_noise = self.edm_preconditioning(sigma, action)
        return self.model(batch, action * c_in, c_noise) * c_out + action * c_skip

    def training_step(self, batch: TensorDict, batch_idx: int) -> Tensor:
        """
        Computes the score matching loss given the perceptual embedding, latent goal, and desired actions.
        """
        batch = self.normalizer(batch)
        batch = self.goal_encoder(batch)
        batch = self.obs_encoder(batch)

        action = batch["action"]

        sigma = self.noise_distribution(shape=(len(action),), device=self.device)
        noise = torch.randn_like(action)
        noised_input = action + noise * unsqueeze_to(sigma, action)

        # We implement the loss with respect to the raw network output as in
        # Equation 8 of https://arxiv.org/pdf/2206.00364. Note that lambda(sigma)
        # is set by the authors to 1/c_out**2, such that each term has an equal
        # weight in the MSE loss.
        c_skip, c_out, c_in, c_noise = self.edm_preconditioning(sigma, action)
        model_output = self.model(batch, noised_input * c_in, c_noise)
        target = (action - c_skip * noised_input) / c_out
        loss = torch.mean(torch.square(model_output - target))

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

        sigmas = self.noise_schedule(self.num_sampling_steps, device=self.device)

        B = batch["obs"].batch_size
        x = torch.randn_like(batch["action"]) * self.sigma_max

        action = self.sampler(
            model=self,  # call self.forward to evaluate the model
            state=batch,
            action=x,
            sigmas=sigmas,
            scaler=None,  # scalar only used for clipping actions
        )

        batch["gt_action"] = batch["action"]
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
            ref_pred = torch.stack([out[:5] for out in prediction.unbind()])
            error = F.mse_loss(ref_pred, batch["ref_action"])
            self.log(
                "only_gripper_val_action_mse", error, batch_size=batch["obs"].shape[0]
            )

            error = torch.mean(torch.square(prediction - batch["gt_action"]))
            self.log("val_action_mse", error, batch_size=batch["obs"].shape[0])

        # return the prediction in case we want to write it back to the environment
        return prediction

    # validation and testing are identical
    test_step = validation_step
