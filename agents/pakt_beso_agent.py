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

from agents.beso_agent import BesoAgent
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


class BesoAgent(BesoAgent):

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
