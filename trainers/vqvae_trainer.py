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
from agents.utils.ema import ExponentialMovingAverage

from trainers.base_trainer import BaseTrainer

log = logging.getLogger(__name__)



class VQVaeTrainer(BaseTrainer):
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
        obs_seq_len: int = 1,
        decay_ema: float = 0.999,
        if_use_ema: bool = False
    ):
        super().__init__(
            trainset,
            valset,
            train_batch_size,
            val_batch_size,
            num_workers,
            device,
            epoch,
            scale_data,
            scaler_type,
            eval_every_n_epochs,
            obs_seq_len,
            decay_ema,
            if_use_ema
        )

    def train(self, agent):
        agent.set_scaler(self.scaler)
         # define optimizer
        if agent.use_lr_scheduler:
            self.optimizer, self.scheduler = agent.configure_optimizers()
        else:
            self.optimizer = agent.configure_optimizers()

        for num_epoch in tqdm(range(self.epoch)):
            epoch_encoder_loss = 0.0
            epoch_vq_loss = 0.0
            epoch_recon_loss = 0.0
            num_batches = 0

            for data in tqdm(self.train_dataloader):
                obs_dict, action, mask = data
                action = self.scaler.scale_output(action)

                (
                    encoder_loss,
                    vq_loss_state,
                    vq_code,
                    vqvae_recon_loss,
                ) = agent.vqvae.vqvae_update(action)  # N T D

                # Accumulate losses
                epoch_encoder_loss += encoder_loss.item()
                epoch_vq_loss += vq_loss_state.item()
                epoch_recon_loss += vqvae_recon_loss
                num_batches += 1

                wandb.log({"pretrain/n_different_codes": len(torch.unique(vq_code))})
                wandb.log(
                    {"pretrain/n_different_combinations": len(torch.unique(vq_code, dim=0))}
                )
                wandb.log({"pretrain/encoder_loss": encoder_loss})
                wandb.log({"pretrain/vq_loss_state": vq_loss_state})
                wandb.log({"pretrain/vqvae_recon_loss": vqvae_recon_loss})

            # Calculate mean losses for the epoch
            mean_encoder_loss = epoch_encoder_loss / num_batches
            mean_vq_loss = epoch_vq_loss / num_batches
            mean_recon_loss = epoch_recon_loss / num_batches

            # Log mean losses for the epoch
            log.info(f"Epoch {num_epoch}: Mean encoder loss: {mean_encoder_loss:.4f}")
            log.info(f"Epoch {num_epoch}: Mean VQ loss: {mean_vq_loss:.4f}")
            log.info(f"Epoch {num_epoch}: Mean reconstruction loss: {mean_recon_loss:.4f}")

            wandb.log({
                "pretrain/epoch_encoder_loss": mean_encoder_loss,
                "pretrain/epoch_vq_loss": mean_vq_loss,
                "pretrain/epoch_recon_loss": mean_recon_loss,
            })

            log.info("training done")
        
        # if num_epoch % 10 == 0:
        #     #todo: save scaler?
        #     state_dict = agent.vqvae.state_dict()
        #     torch.save(state_dict, os.path.join(save_path, "trained_vqvae.pt"))
        return agent.vqvae.state_dict()


    def evaluate_nsteps(self, VQVAEmodel, criterion, loader, step_id, val_iters, split='val'):
        pass
