import logging
import os

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
import hydra
from typing import Optional
from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class BC_Agent(BaseAgent):
    def __init__(
        self,
        model: DictConfig,
        obs_encoders: DictConfig,
        language_encoders: DictConfig,
        optimization: DictConfig,
        action_seq_size: int,
        if_robot_states: bool = False,
        if_film_condition: bool = False,
        device: str = "cpu",
        state_dim: int = 7,
        latent_dim: int = 64,
        multistep: int = 10,
    ):
        super().__init__(
            model=model,
            obs_encoders=obs_encoders,
            language_encoders=language_encoders,
            device=device,
            state_dim=state_dim,
            latent_dim=latent_dim,
            multistep=multistep
        )

        # self.img_encoder = hydra.utils.instantiate(obs_encoders).to(device)
        # self.model = hydra.utils.instantiate(model).to(device)
        # self.state_emb = nn.Linear(state_dim, latent_dim)

        self.if_robot_states = if_robot_states
        self.if_film_condition = if_film_condition

        self.eval_model_name = "eval_best_bc.pth"
        self.last_model_name = "last_bc.pth"

        self.action_seq_size = action_seq_size

        self.optimizer_config = optimization
        self.use_lr_scheduler = False

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.optimizer_config, params=self.parameters()
        )
        return optimizer

    def forward(self, obs_dict, actions=None):

        # with torch.no_grad():
        perceptual_emb, latent_goal = self.compute_input_embeddings(obs_dict)

        # shape of perceptural_emb is torch.Size([64, 1, 256])
        # make prediction
        pred = self.model(
            perceptual_emb,
            latent_goal
        )

        if self.training and actions is not None:
            loss = F.mse_loss(pred, actions)

            return loss

        return pred

