from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Callable, Type

import lightning as L
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from agents.utils.scaler import Scaler
    from environments.dataset.base_dataset import TrajectoryDataset


log = logging.getLogger(__name__)


class BaseAgent(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        obs_encoder: Callable[[TrajectoryDataset], nn.Module],
        scaler: Type[Scaler],
        language_encoder: nn.Module,
        latent_dim: int,  # BALAZS: add state_encoder to hydra
        obs_seq_len: int,
        act_seq_len: int,
        dataset: TrajectoryDataset,
    ):
        super().__init__()

        self.model = model
        self.obs_encoder = obs_encoder(dataset)
        self.scaler = scaler(dataset.all_actions)
        self.language_encoder = language_encoder

        state_dim = dataset.state_dim
        self.state_emb = nn.Linear(state_dim, latent_dim)

        self.camera_names = dataset.camera_names

        # for inference
        self.rollout_step_counter = 0
        self.act_seq_len = act_seq_len
        self.obs_seq_len = obs_seq_len

        self.obs_seq: dict[str, deque[torch.Tensor]] = {}

    def encode_obs(self, obs_dict):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        #########################################
        # deal with language embedding
        #########################################
        if "lang" in obs_dict:
            obs_dict["lang_emb"] = self.language_encoder(obs_dict["lang"]).float()

        latent_goal = obs_dict["lang_emb"]

        if self.if_film_condition:
            obs_embedding = self.obs_encoder(obs_dict, latent_goal)
        else:
            # obs_dict is a dict with two images and one lang: images are [64,3,256,256]
            # obs_embedding has shape [B, N, E], where N is number of cameras and E is embedding dim
            obs_embedding = self.obs_encoder(obs_dict)

        #########################################
        # add robot states
        #########################################
        if self.if_robot_states and "robot_states" in obs_dict.keys():
            robot_states = obs_dict["robot_states"]
            robot_states = self.state_emb(robot_states)

            obs_embedding = torch.cat([obs_embedding, robot_states], dim=1)

        return obs_embedding, latent_goal
