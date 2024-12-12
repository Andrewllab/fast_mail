"""Shared utilities for all main scripts."""

import os
import pickle
import random
import logging
import wandb
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from agents.utils.scaler import Scaler, ActionScaler, MinMaxScaler

log = logging.getLogger(__name__)


class BaseTrainTester:
    """Basic train/test class to be inherited."""

    def __init__(
            self,
            trainset: DictConfig,
            valset: DictConfig,
            train_batch_size: int = 512,
            val_batch_size: int = 512,
            num_workers: int = 8,
            device: str = 'cpu',
            epoch: int = 100,
            scale_data: bool = True,
            scaler_type: str = None,
            eval_every_n_epochs: int = 50,
            obs_seq_len: int = 1
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

        self.obs_seq_len = obs_seq_len

        self.eval_every_n_epochs = eval_every_n_epochs

        self.epoch = epoch

        self.device = device
        self.working_dir = os.getcwd()

        self.scaler_type = scaler_type

        if self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler(self.trainset.get_all_actions(), scale_data, device)
        else:
            self.scaler = ActionScaler(self.trainset.get_all_actions(), scale_data, device)

    def main(self, agent):
        """Run main training/testing pipeline."""

        for num_epoch in tqdm(range(self.epoch)):

            epoch_loss = torch.tensor(0.0).to(self.device)

            for data in self.train_dataloader:
                obs_dict, action, mask = data

                # put data on cuda
                for camera in obs_dict.keys():
                    obs_dict[camera] = obs_dict[camera].to(self.device)

                    if 'rgb' not in camera:
                        continue
                    obs_dict[camera] = obs_dict[camera][:, :self.obs_seq_len].contiguous()

                action = self.scaler.scale_output(action)
                action = action[:, self.obs_seq_len - 1:, :].contiguous()

                batch_loss = self.train_one_step(agent, obs_dict, action)

                epoch_loss += batch_loss

            epoch_loss = epoch_loss / len(self.train_dataloader)

            wandb.log({"train_loss": epoch_loss.item()})
            log.info("Epoch {}: Mean train loss is {}".format(num_epoch, epoch_loss.item()))

        log.info("training done")
        agent.store_model_weights(agent.working_dir, sv_name='last_model.pth')

    def train_one_step(self, agent, obs_dict, action):
        """Run a single training step."""
        agent.train()

        loss = agent.train_one_step(obs_dict, action)

        agent.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        agent.optimizer.step()

        return loss

    @torch.no_grad()
    def evaluate_nsteps(self, model, criterion, loader, step_id, val_iters,
                        split='val'):
        """Run a given number of evaluation steps."""
        return None

