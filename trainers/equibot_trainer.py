"""Shared utilities for all main scripts."""

import logging
import os
from pathlib import Path

import einops
import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.camera_utils import quat2Mat, quat2MatBatch

log = logging.getLogger(__name__)


class EquiBotTrainer:
    """Basic train/test class to be inherited."""

    def __init__(
        self,
        trainset: DictConfig,
        valset: DictConfig,
        use_lr_scheduler: bool,
        train_batch_size: int = 512,
        val_batch_size: int = 512,
        num_workers: int = 8,
        device: str = "cpu",
        epoch: int = 100,
        obs_seq_len: int = 1,
    ):
        """Initialize."""

        self.trainset = hydra.utils.instantiate(trainset)
        # self.valset = hydra.utils.instantiate(valset)

        self.train_dataloader = DataLoader(
            self.trainset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # self.test_dataloader = DataLoader(
        #     self.valset,
        #     batch_size=val_batch_size,
        #     shuffle=False,
        #     num_workers=0,
        #     pin_memory=True,
        #     drop_last=False
        # )

        self.batch_size = train_batch_size
        self.obs_seq_len = obs_seq_len
        self.epoch = epoch
        self.device = device
        self.working_dir = os.getcwd()
        self.use_lr_scheduler = use_lr_scheduler

    def main(self, agent):
        """Run main training/testing pipeline."""

        self.optimizer, self.scheduler = agent.configure_optimizers(
            num_training_steps=self.epoch * len(self.trainset) // self.batch_size
        )

        for num_epoch in tqdm(range(self.epoch)):
            epoch_loss = torch.tensor(0.0).to(self.device)
            for data in tqdm(self.train_dataloader):
                obs_dict, action, mask = data

                gravity_dir = einops.repeat(
                    torch.tensor([0, 0, -1]),
                    "d -> b t d",
                    b=action.shape[0],
                    t=self.obs_seq_len,
                )

                obs_dict = {
                    "pc": obs_dict["sampled_point_cloud"][:, : self.obs_seq_len, :, :3],
                    "robot_states": torch.cat(
                        [
                            obs_dict["eef_pos"][:, : self.obs_seq_len],
                            obs_dict["eef_rot"][:, : self.obs_seq_len],
                            gravity_dir,
                            obs_dict["gripper_state"][:, : self.obs_seq_len],
                        ],
                        dim=-1,
                    ),
                }

                # put data on cuda
                for camera in obs_dict.keys():
                    if camera == "lang":
                        continue

                    obs_dict[camera] = obs_dict[camera].to(self.device).contiguous()

                action = (
                    action[:, self.obs_seq_len - 1 :, :].to(self.device).contiguous()
                )

                batch_loss = self.train_one_step(agent, obs_dict, action)

                epoch_loss += batch_loss

            if num_epoch % 20 == 0:
                agent.save_snapshot(Path(self.working_dir) / f"epoch_{num_epoch}.pth")

            epoch_loss = epoch_loss / len(self.train_dataloader)

            wandb.log({"epoch_train_loss": epoch_loss.item()})
            log.info(
                "Epoch {}: Mean train loss is {}".format(num_epoch, epoch_loss.item())
            )

        agent.save_snapshot(Path(self.working_dir) / "last_model.pth")
        log.info("training done")

    def train_one_step(self, agent, obs_dict, action):
        """Run a single training step."""
        agent.train()

        metrics = agent(obs_dict, action)

        wandb.log({k: v.item() for k, v in metrics.items()})

        self.optimizer.zero_grad(set_to_none=True)
        metrics["loss"].backward()
        self.optimizer.step()

        if self.use_lr_scheduler:
            self.scheduler.step()

        return metrics["loss"]

    @torch.no_grad()
    def evaluate_nsteps(
        self, model, criterion, loader, step_id, val_iters, split="val"
    ):
        """Run a given number of evaluation steps."""
        return None
