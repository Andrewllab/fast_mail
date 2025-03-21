from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

    from agents.encoders.encoder import ObservationEncoder
    from environments.dataset.base_dataset import TrajectoryDataset

logger = logging.getLogger(__name__)


# Non-diffusion based decoder-only model
class Dec_only(nn.Module):
    def __init__(
        self,
        encoder: DictConfig,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        device: str,
        goal_conditioned: bool,
        embed_dim: int,
        embed_pdrob: float,
        goal_seq_len: int,
        obs_seq_len: int,
        action_seq_len: int,
        linear_output: bool = False,
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder)

        self.device = device

        # mainly used for language condition or goal image condition
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0

        # the seq_size is the number of tokens in the input sequence
        self.seq_size = goal_seq_len + obs_seq_len + action_seq_len

        # linear embedding for the state
        self.tok_emb = nn.Linear(state_dim, embed_dim)

        # linear embedding for the goal
        self.goal_emb = nn.Linear(goal_dim, embed_dim)

        # position embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # get an action embedding
        self.query_embed = nn.Embedding(action_seq_len, embed_dim)

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100), nn.GELU(), nn.Linear(100, self.action_dim)
            )
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        # logger.info(
        #     "number of parameters: %e", sum(p.numel() for p in self.parameters())
        # )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        states,
        goals=None,
    ):

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        b, t, dim = states.size()

        if self.goal_conditioned:
            goal_embed = self.goal_emb(goals)
            goal_x = self.drop(goal_embed + self.pos_emb[:, : self.goal_seq_len, :])

        state_embed = self.tok_emb(states)
        state_x = self.drop(
            state_embed
            + self.pos_emb[:, self.goal_seq_len : (self.goal_seq_len + t), :]
        )

        action_seq = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)

        if self.goal_conditioned:
            input_seq = torch.cat([goal_x, state_x, action_seq], dim=1)
        else:
            input_seq = torch.cat([state_x, action_seq], dim=1)

        encoder_output = self.encoder(input_seq)

        pred_actions = self.action_pred(encoder_output[:, -self.action_seq_len :, :])

        return pred_actions


# Diffusion based decoder-only model, we need time embedding and noisy antions inputs here
class Noise_Dec_only(nn.Module):
    def __init__(
        self,
        decoder: Module,
        time_encoder: Callable[[int, int], Module] | None,
        sigma_encoder: Callable[[int], Module],
        action_head: Callable[[int, int], Module],
        dataset: TrajectoryDataset,
        obs_encoder: ObservationEncoder,
        embed_dim: int,
        embed_pdrob: float,
    ):
        super().__init__()

        self.decoder = decoder
        self.sigma_encoder = sigma_encoder(embed_dim)
        self.action_head = action_head(embed_dim, dataset.action_dim)

        # we use time to refer to the position in the sequence of tokens
        # this often corresponds to real time, but not always, e.g. with goal tokens
        if time_encoder is not None:
            self.obs_time_encoder = time_encoder(obs_encoder.obs_seq_len, embed_dim)
            self.action_time_encoder = time_encoder(dataset.action_seq_len, embed_dim)

            if self.dataset.goal_seq_len > 0:
                self.goal_time_encoder = time_encoder(dataset.goal_seq_len, embed_dim)
        else:
            self.obs_time_encoder = time_encoder
            self.action_time_encoder = time_encoder
            self.goal_time_encoder = time_encoder

        # linear embedding for the state
        self.state_encoder = nn.LazyLinear(embed_dim)

        # linear embedding for the goal
        self.goal_encoder = nn.LazyLinear(embed_dim)

        # linear embedding for the action
        self.action_encoder = nn.LazyLinear(embed_dim)

        self.drop = nn.Dropout(embed_pdrob)

        self.action_seq_len = dataset.action_seq_len

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self, states: Tensor, actions: Tensor, goal: Tensor | None, sigma: Tensor
    ) -> Tensor:

        input_seq = []

        input_seq.append(self.sigma_encoder(sigma))

        if goal is not None:
            goal_embed = self.goal_encoder(goal)
            if self.goal_time_encoder is not None:
                indices = torch.arange(goal_embed.shape[-2], dtype=torch.long)
                goal_embed += self.goal_time_encoder(indices)
            input_seq.append(self.drop(goal_embed))

        state_embed = self.state_encoder(states)
        if self.obs_time_encoder is not None:
            indices = torch.arange(state_embed.shape[-2], dtype=torch.long)
            state_embed += self.obs_time_encoder(indices)
        input_seq.append(self.drop(state_embed))

        action_embed = self.action_encoder(actions)
        if self.action_time_encoder is not None:
            indices = torch.arange(action_embed.shape[-2], dtype=torch.long)
            action_embed += self.action_time_encoder(indices)
        input_seq.append(self.drop(action_embed))

        input_seq = torch.cat(input_seq, dim=1)

        output = self.decoder(input_seq)

        pred_actions = self.action_head(output[:, -self.action_seq_len :])

        return pred_actions
