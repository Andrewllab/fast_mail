from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module

from environments.specs import DataSpecs, EmbedSpec
from utils.nested import cat_nested, make_jagged_nested_tensors_compatible

log = logging.getLogger(__name__)


class DecoderOnlyNoise(nn.Module):
    """Decoder-only model for noise prediction in the diffusion process.

    This model takes observation embeddings, action tokens, goal tokens (if
    available), and sigma as input. The actions and goal tokens are passed
    through a linear embedding layer and a learned token position encoder is
    added. The observation embedding is not transformed, and is only given a
    learned positional encoding if the sequence has a fixed length and
    `time_encode_obs` is set to True.
    """

    def __init__(
        self,
        specs: DataSpecs,
        decoder: Module,
        seq_position_encoder: Callable[[int, int], Module] | None,
        sigma_encoder: Callable[[int], Module],
        action_head: Callable[[int, int], Module],
        dropout_prob: float,
        action_obs_tokenizer: Callable[[TensorDict], TensorDict] | None = None,
        time_encode_obs: bool = True,
    ):
        super().__init__()

        if time_encode_obs:
            log.warning(
                "`time_encode_obs` is set to True, so the decoder will add positional encodings to the observation embeddings. Make sure that the observation embeddings do not already contain positional encodings."
            )

        self.action_obs_tokenizer = None
        if action_obs_tokenizer is not None:
            self.action_obs_tokenizer = action_obs_tokenizer(specs=specs)
            specs = self.action_obs_tokenizer.specs
        token_dim = specs.obs_embed_dim

        self.decoder = decoder(token_dim)
        self.sigma_encoder = sigma_encoder(token_dim)
        self.action_head = action_head(token_dim, specs.action_dim)

        log.debug(
            f"Noise model expects to receive {specs.obs_embed_seq_len if specs.obs_embed_seq_len is not None else 'a variable number of'} obs embedding tokens across all time steps."
        )

        # we use time to refer to the position in the sequence of tokens
        # this often corresponds to real time, but not always, e.g. with goal tokens
        if seq_position_encoder is not None:
            embed_spec = specs.obs["embed"]
            assert isinstance(embed_spec, EmbedSpec)
            if embed_spec.fixed_shape and time_encode_obs:
                self.obs_pos_encoder = seq_position_encoder(
                    embed_spec.n_tokens, token_dim
                )
            else:
                self.obs_pos_encoder = None

            self.action_pos_encoder = seq_position_encoder(
                specs.action_seq_len, token_dim
            )

            if specs.goal_embed_seq_len is not None:
                self.goal_pos_encoder = seq_position_encoder(
                    specs.goal_embed_seq_len, token_dim
                )
        else:
            self.obs_pos_encoder = None
            self.action_pos_encoder = None
            self.goal_pos_encoder = None

        # linear embedding for the action
        self.action_encoder = nn.Linear(specs.action_dim, token_dim)

        # linear embedding for the goal
        if specs.goal_embed_dim is not None:
            self.goal_encoder = nn.Linear(specs.goal_embed_dim, token_dim)

        if dropout_prob > 0:
            self.drop = nn.Dropout(dropout_prob)
        else:
            self.drop = nn.Identity()

        self.action_seq_len = specs.action_seq_len

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, batch: TensorDict, actions: Tensor, sigma: Tensor) -> Tensor:

        input_seq = []

        input_seq.append(self.sigma_encoder(sigma).unsqueeze(dim=-2))

        # === GOAL TOKENS ===
        goal = batch.get(("goal", "embed"), None)
        if goal is not None:
            goal_embed = self.goal_encoder(goal)
            if self.goal_pos_encoder is not None:
                indices = torch.arange(
                    goal_embed.shape[1], dtype=torch.long, device=goal_embed.device
                )
                goal_embed += self.goal_pos_encoder(indices)
            input_seq.append(self.drop(goal_embed))

        # === ACTION TOKENS ===
        if self.action_obs_tokenizer is not None:
            batch["noisy_action"] = actions
            batch, action_embed = self.action_obs_tokenizer(batch)
        else:
            action_embed = self.action_encoder(actions)

        if self.action_pos_encoder is not None:
            indices = torch.arange(
                action_embed.shape[1], dtype=torch.long, device=action_embed.device
            )
            action_embed += self.action_pos_encoder(indices)

        # === OBSERVATION TOKENS ===
        obs_embed = batch["obs", "embed"]
        if self.obs_pos_encoder is not None:
            assert not obs_embed.is_nested
            indices = torch.arange(
                obs_embed.shape[1], dtype=torch.long, device=obs_embed.device
            )
            obs_embed += self.obs_pos_encoder(indices)

        # Combine all input tokens
        input_seq.append(self.drop(obs_embed))
        input_seq.append(self.drop(action_embed))

        input_seq = cat_nested(input_seq, dim=1)

        # Actual forward pass through the decoder
        output = self.decoder(input_seq)

        # retrieve the decoded action tokens from the sequence
        if output.is_nested:
            if not action_embed.is_nested:
                action_tokens = [out[-self.action_seq_len :] for out in output.unbind()]
            else:
                num_elements = torch.diff(action_embed.offsets())
                action_tokens = [
                    out[-num_elements[i] :] for i, out in enumerate(output.unbind())
                ]
                action_tokens = torch.nested.as_nested_tensor(
                    action_tokens, layout=torch.jagged
                )
        else:
            action_tokens = output[:, -self.action_seq_len :]

        pred_actions = self.action_head(action_tokens)

        if pred_actions.is_nested:
            _, pred_actions = make_jagged_nested_tensors_compatible(
                actions, pred_actions
            )

        return pred_actions
