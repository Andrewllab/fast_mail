from __future__ import annotations

import logging
from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from environments.specs import DataSpecs, EmbedSpec
from transforms.base_transform import KeyMapping, Transform
from utils.nested import cat_nested

log = logging.getLogger(__name__)


class RobotStateEncoder(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        model: Callable[[int, int], Module],
        embed_dim: int,
        obs_key: str | Sequence[str] = "robot_state",
        token_pos_encoder: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()

        obs_keys = [obs_key] if isinstance(obs_key, str) else list(obs_key)

        input_specs = []
        for key in obs_keys:
            if key not in specs.obs:
                raise KeyError(f"Observation spec at specs.obs[{key}] not found.")

            input_specs.append(specs.obs[key])
            if len(specs.obs[key].shape) != 2:
                raise ValueError(
                    f"Observation spec at specs.obs[{key}] must be of shape [T,M]"
                )

        times = [spec.shape[0] for spec in input_specs]
        if not all(t == times[0] for t in times):
            raise ValueError(
                f"All observation specs must have the same time dimension, "
                f"but got {[spec.shape[0] for spec in input_specs]}"
            )
        time = times[0]

        state_dim = sum(spec.shape[1] for spec in input_specs)

        # instantiate the model
        self.model = model(state_dim, embed_dim)

        # create encoder for token position
        if token_pos_encoder is None:
            log.warning("No token position encoder provided. Using nn.Embedding.")
            token_pos_encoder = nn.Embedding
        self.token_pos_encoder = token_pos_encoder(time, embed_dim)

        # create a modified specs object for the output
        # each time step produces a single token
        new_spec = EmbedSpec(embed_dim=embed_dim, n_tokens=time)
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        if "embed" in obs_specs:
            # if embedding sequence has fixed length, increase length to account for state tokens
            embed_spec = obs_specs["embed"]
            assert isinstance(embed_spec, EmbedSpec)
            new_spec = embed_spec.concat(new_spec)
        obs_specs["embed"] = new_spec
        self._output_specs = specs.replace(obs=obs_specs)

        self._obs_keys = obs_keys

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=[("obs", "embed")] + [("obs", key) for key in self._obs_keys],
                out_keys=[("obs", "embed")],
            )
        ]

    def _call_one(self, obs_embed: Tensor | None, *robot_state: Tensor) -> Tensor:
        if len(robot_state) > 1:
            robot_state = torch.cat(robot_state, dim=-1)
        else:
            robot_state = robot_state[0]

        # (B, T, M) -> (B, T, D)
        features = self.model(robot_state)

        # add encoding of the token position to each token
        N = features.shape[1]
        token_indices = torch.arange(N, dtype=torch.long, device=features.device)
        token_pos_embed = self.token_pos_encoder(token_indices)
        features += token_pos_embed

        if obs_embed is None:
            return features

        # concatenate along N dimension of embedding
        # obs_embed: (B, N, D)
        return cat_nested([obs_embed, features], dim=-2)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
