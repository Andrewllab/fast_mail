import logging
import os
from collections import deque

import einops
import cv2
import ast
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from typing import Optional
from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class BC_Agent(BaseAgent):
    def __init__(
            self,
            model: DictConfig,
            optimization: DictConfig,
            trainset: DictConfig,
            valset: DictConfig,
            train_batch_size,
            val_batch_size,
            num_workers,
            device: str,
            epoch: int,
            obs_seq_len: int,
            action_seq_size: int,
            scale_data,
            eval_every_n_epochs: int = 50
    ):
        super().__init__(model=model, trainset=trainset, valset=valset, train_batch_size=train_batch_size,
                         val_batch_size=val_batch_size, num_workers=num_workers, device=device,
                         epoch=epoch, scale_data=scale_data, eval_every_n_epochs=eval_every_n_epochs)

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.parameters()
        )

        self.eval_model_name = "eval_best_bc.pth"
        self.last_model_name = "last_bc.pth"

        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(self.device)
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(self.device)

        self.obs_seq_len = obs_seq_len
        self.action_seq_size = action_seq_size

        self.rollout_step_counter = 0
        self.multistep = action_seq_size

        self.train_loss = []
        self.test_mse = []

    def train_agent(self):

        for data in self.train_dataloader:
            obs_dict, action, mask = data

            for camera in obs_dict.keys():
                obs_dict[camera] = obs_dict[camera].to(self.device)
                # if 'rgb' not in camera:
                #     continue
                # obs_dict[camera] = obs_dict[camera][:, :self.obs_seq_len].contiguous()

            action = self.scaler.scale_output(action)
            action = action[:, self.obs_seq_len - 1:, :].contiguous()

            batch_loss = self.train_step(obs_dict, action)

            wandb.log({"train_loss": batch_loss.item()})

    def train_step(self, state, actions: torch.Tensor, goal: Optional[torch.Tensor] = None):
        """
        Executes a single training step on a mini-batch of data
        """
        self.model.train()

        out = self.model(state)
        loss = F.mse_loss(out, actions)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss

    @torch.no_grad()
    def evaluate(self, state, action: torch.Tensor, goal: Optional[torch.Tensor] = None):
        """
        Method for evaluating the model on one epoch of data
        """
        self.model.eval()

        total_mse = 0.0

        if goal is not None:
            goal = self.scaler.scale_input(goal)
            out = self.model(torch.cat([state, goal], dim=-1))
        else:
            out = self.model(state)

        mse = F.mse_loss(out, action)  # , reduction="none")
        total_mse += mse.mean(dim=-1).sum().item()
        return total_mse

    def reset(self):
        """ Resets the context of the model."""
        self.rollout_step_counter = 0

    @torch.no_grad()
    def predict(self, obs, goal: Optional[torch.Tensor] = None) -> torch.Tensor:
        imgs_seq = {}

        for camera, data in obs.items():
            if 'rgb' not in camera:
                continue

            obs[camera] = torch.from_numpy(data).to(self.device).float().permute(2, 0, 1).unsqueeze(0) / 255.

            imgs_seq[camera] = obs[camera].unsqueeze(0)

        imgs_seq['lang_emb'] = obs['lang_emb'].to(self.device).unsqueeze(0)

        # imgs_seq['robot_states'] = torch.from_numpy(obs['robot_states']).to(self.device).float().unsqueeze(0).unsqueeze(0)

        if self.rollout_step_counter % self.multistep == 0:
            self.model.eval()

            # predict action sequence
            pred_action_seq = self.model(imgs_seq)

            pred_action_seq = self.scaler.inverse_scale_output(pred_action_seq)

            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]

        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')

        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0

        return current_action.detach().cpu().numpy()

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """

        if sv_name is None:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, "model_state_dict.pth")))
        else:
            self.model.load_state_dict(torch.load(os.path.join(weights_path, sv_name)))
        log.info('Loaded pre-trained model parameters')

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """

        if sv_name is None:
            torch.save(self.model.state_dict(), os.path.join(store_path, "model_state_dict.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(store_path, sv_name))