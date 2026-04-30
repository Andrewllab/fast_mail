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
from utils.math import quat_from_axis_angle, quat_rotation_distance
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
        # weight decays
        qk_norm_weight_decay: float | None = None,  # <-- new
        exclude_norms_from_weight_decay: bool = True,  # <-- new
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
            qk_norm_weight_decay=qk_norm_weight_decay,
            exclude_norms_from_weight_decay=exclude_norms_from_weight_decay,
        )
        self.num_timesteps = specs.action_seq_len

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # values logged here get averaged over an epoch instead of over a step

        # Log info from completed episodes, if validating on environment
        # rollouts.
        # Do this before predicting the next action, just in case.

        if "done" in batch and torch.any(batch["done"]) and "episode_info" in batch:
            done = batch["done"]
            episode_info = batch["episode_info"]
            completed_episodes = episode_info[done]
            # lightning expects scalar values that have already been reduced
            metrics = completed_episodes.float().mean().to_dict()

            assert self.checkpoint_metadata and "epoch" in self.checkpoint_metadata
            self.log_dict(
                {"ckpt_epoch": self.checkpoint_metadata["epoch"], **metrics},
                # since the number of completed episodes can vary between steps,
                # we need to provide lightning with the size of this "batch" of
                # completed episodes for correct averaging
                batch_size=done.sum().item(),
            )

        prediction = self.predict_step(batch, batch_idx)

        if "ref_action" in batch:
            error = F.mse_loss(prediction, batch["ref_action"])
            self.log("val_action_mse", error, batch_size=batch["obs"].shape[0])

            pred_pos = prediction[..., :3]
            ref_pos = batch["ref_action"][..., :3]
            position_error = F.mse_loss(pred_pos, ref_pos)
            self.log(
                "val_position_mse", position_error, batch_size=batch["obs"].shape[0]
            )

            if (
                prediction.shape[-1] == 7
            ):  # if the action has 7 dimensions the rotation is a axis angle
                pred_rot = prediction[..., 3:6]
                pred_quat = quat_from_axis_angle(pred_rot)

                ref_rot = batch["ref_action"][..., 3:6]
                ref_quat = quat_from_axis_angle(ref_rot)
            else:
                pred_quat = prediction[..., 3:7]
                ref_quat = batch["ref_action"][..., 3:7]

            rotation_error = quat_rotation_distance(pred_quat, ref_quat).mean()
            self.log(
                "val_rotation_mse", rotation_error, batch_size=batch["obs"].shape[0]
            )

            pred_gripper = prediction[..., -1:]
            ref_gripper = batch["ref_action"][..., -1:]
            gripper_error = F.mse_loss(pred_gripper, ref_gripper)
            self.log("val_gripper_mse", gripper_error, batch_size=batch["obs"].shape[0])

        # return the prediction in case we want to write it back to the environment
        return prediction

    # validation and testing are identical
    test_step = validation_step
