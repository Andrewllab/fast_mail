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
        # keypoint loss dropout
        keypoint_loss_dropout: float = 0.0,  # <-- new
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
        self.keypoint_loss_dropout = keypoint_loss_dropout

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
        c_skip, c_out, c_in, c_noise = self.get_preconditioning_factors(sigma, action)
        model_output = self.model(batch, noised_input * c_in, c_noise)
        target = (action - c_skip * noised_input) / c_out
        if model_output.is_nested:
            loss = self.dropout_keypoint_loss_nested(model_output, target, batch)
        else:
            loss = self.dropout_keypoint_loss(model_output, target, batch)

        # log these values per step and per epoch
        log_dict = {
            "loss": loss,
        }
        if self._lr_scheduler_func is not None:
            log_dict["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log_dict(
            log_dict,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.shape[0],
        )

        return loss

    def dropout_keypoint_loss_nested(
        self, model_output: Tensor, target: Tensor, batch: TensorDict
    ) -> Tensor:
        if self.keypoint_loss_dropout <= 0.0:
            return F.mse_loss(model_output.values(), target.values())

        action_mask = (
            batch["obs"]["action_points"]["gripper_ids"] != 0
        )  # nested (B, N_i)

        B = action_mask.size(0)
        offsets = action_mask.offsets()  # (B+1,)
        lengths = offsets[1:] - offsets[:-1]  # (B,)

        keep_keypoints = (
            torch.rand(B, device=action_mask.device) > self.keypoint_loss_dropout
        )  # (B,)

        # Broadcast keep_keypoints to per-point via repeat_interleave on the flat values
        keep_per_point = keep_keypoints.repeat_interleave(lengths)  # (sum N_i,)
        action_mask_flat = action_mask.values()  # (sum N_i,)
        point_mask_flat = (action_mask_flat | keep_per_point).float()  # (sum N_i,)

        # Per-element squared error on the flat values tensor — avoids nested ops in autograd
        sq_err_flat = (model_output.values() - target.values()).pow(
            2
        )  # (sum N_i, D...)
        while sq_err_flat.dim() > 1:
            sq_err_flat = sq_err_flat.mean(dim=-1)
        # sq_err_flat is now (sum N_i,)

        masked = sq_err_flat * point_mask_flat  # (sum N_i,)

        # Per-sample sums via index_add — well-supported backward
        row_ids = torch.repeat_interleave(
            torch.arange(B, device=offsets.device), lengths
        )  # (sum N_i,)
        per_sample_sum = torch.zeros(B, device=masked.device, dtype=masked.dtype)
        per_sample_sum = per_sample_sum.index_add(0, row_ids, masked)  # (B,)

        counts = point_mask_flat.new_zeros(B).index_add(0, row_ids, point_mask_flat)
        counts = counts.clamp(min=1.0)

        per_sample_loss = per_sample_sum / counts  # (B,)
        return per_sample_loss.mean()

    def dropout_keypoint_loss(
        self, model_output: Tensor, target: Tensor, batch: TensorDict
    ) -> Tensor:
        if self.keypoint_loss_dropout <= 0.0:
            return F.mse_loss(model_output, target)

        # True for action points, False for keypoints
        action_mask = batch["obs"]["action_points"]["gripper_ids"] != 0  # (B, N)

        # Per-sample: True with prob (1 - p_drop) = "keep keypoints this step"
        keep_keypoints = (
            torch.rand(action_mask.shape[0], device=model_output.device)
            > self.keypoint_loss_dropout
        )  # (B,)

        # Final mask: action points always in, keypoints in only when kept
        point_mask = action_mask | keep_keypoints[:, None]  # (B, N)

        # Per-element squared error, then mean per sample over included points
        sq_err = (model_output - target).pow(2)  # (B, N, D) or (B, N, ...)
        # reduce over feature dims if any
        while sq_err.dim() > 2:
            sq_err = sq_err.mean(dim=-1)
        # now sq_err is (B, N)

        point_mask_f = point_mask.float()
        counts = point_mask_f.sum(dim=1)  # (B,)

        # avoid div-by-zero (shouldn't happen since action points are always kept)
        counts = counts.clamp(min=1.0)
        per_sample_loss = (sq_err * point_mask_f).sum(dim=1) / counts  # (B,)

        return per_sample_loss.mean()

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
