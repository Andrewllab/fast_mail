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
            optimization: DictConfig,
            action_seq_size: int,
            if_robot_states: bool = False,
            if_film_condition: bool = False,
            device: str = 'cpu',
            state_dim: int = 7,
            latent_dim: int = 64
    ):
        super().__init__(device=device)

        self.img_encoder = hydra.utils.instantiate(obs_encoders).to(device)
        self.model = hydra.utils.instantiate(model).to(device)

        self.if_robot_states = if_robot_states
        self.if_film_condition = if_film_condition
        self.state_emb = nn.Linear(state_dim, latent_dim)

        self.eval_model_name = "eval_best_bc.pth"
        self.last_model_name = "last_bc.pth"

        self.action_seq_size = action_seq_size

        self.rollout_step_counter = 0
        self.multistep = action_seq_size

        self.optimizer_config = optimization
        self.use_lr_scheduler = False

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.optimizer_config, params=self.parameters())
        return optimizer

    def compute_input_embeddings(self, obs_dict):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """

        latent_goal = obs_dict['lang_emb']

        # print(f"the shape of this dict is {obs_dict[list(obs_dict.keys())[0]].shape}")
        B, T, C, H, W = obs_dict[list(obs_dict.keys())[0]].shape

        for camera in obs_dict.keys():
            if 'rgb' not in camera:
                continue
            # print(obs_dict[camera].shape)
            obs_dict[camera] = obs_dict[camera].view(B * T, C, H, W)
        # print(self.if_film_condition)
        if self.if_film_condition:
            perceptual_emb = self.img_encoder(obs_dict, latent_goal)
        else:
            # obs_dict is a dict with two images and one lang: images are [64,3,256,256]
            perceptual_emb = self.img_encoder(obs_dict)

        if self.if_robot_states and "robot_states" in obs_dict.keys():
            robot_states = obs_dict['robot_states']
            robot_states = self.state_emb(robot_states)

            perceptual_emb = torch.cat([perceptual_emb, robot_states], dim=1)

        return perceptual_emb, latent_goal

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

        if self.if_robot_states and "robot_states" in obs.keys():
            imgs_seq['robot_states'] = torch.from_numpy(obs['robot_states']).to(self.device).float().unsqueeze(0).unsqueeze(0)

        if self.rollout_step_counter % self.multistep == 0:
            self.eval()

            # predict action sequence
            pred_action_seq = self(imgs_seq)
            pred_action_seq = self.scaler.inverse_scale_output(pred_action_seq)
            self.pred_action_seq = pred_action_seq

        current_action = self.pred_action_seq[0, self.rollout_step_counter]

        if len(current_action.shape) == 2:
            current_action = einops.rearrange(current_action, 'b d -> b 1 d')

        self.rollout_step_counter += 1
        if self.rollout_step_counter == self.multistep:
            self.rollout_step_counter = 0

        return current_action.detach().cpu().numpy()
