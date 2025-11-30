from __future__ import annotations

import logging
from typing import Callable

import torch.nn as nn
from torch import Tensor

from environments.specs import DataSpecs, EmbedSpec
from models.transformer import AttentionPoolingLayer, TransformerEncoder
from transforms.base_transform import KeyMapping, Transform

log = logging.getLogger(__name__)


class AttentionPoolingTokenizer(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        embed_dim: int,
        n_layers: int,
        n_tokens: int,
        norm: type[nn.Module],
        attention: type[nn.Module],
        residual_dropout: float | Callable[[], nn.Module],
        mlp1: type[nn.Module],
        activation: type[nn.Module] | str,
        mlp_dropout: float | Callable[[], nn.Module],
        mlp2: type[nn.Module],
        norm_first: bool,
        token_pos_encoder: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()

        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            n_layers=n_layers,
            norm=norm,
            attention=attention,
            residual_dropout=residual_dropout,
            mlp1=mlp1,
            activation=activation,
            mlp_dropout=mlp_dropout,
            mlp2=mlp2,
            norm_first=norm_first,
        )

        self.pooling = AttentionPoolingLayer(
            embed_dim=embed_dim,
            n_tokens=n_tokens,
            norm=norm,
            attention=attention,
            residual_dropout=residual_dropout,
            mlp1=mlp1,
            activation=activation,
            mlp_dropout=mlp_dropout,
            mlp2=mlp2,
            norm_first=norm_first,
        )
        # TransformerEncoder uses a norm after each layer
        self.final_norm = norm(embed_dim)

        # create encoder for token position
        if token_pos_encoder is None:
            log.warning("No token position encoder provided. Using nn.Embedding.")
            token_pos_encoder = nn.Embedding
        self.token_pos_encoder = token_pos_encoder(n_tokens, embed_dim)

        # create a modified specs object for the output
        new_spec = EmbedSpec(embed_dim=embed_dim, n_tokens=n_tokens)
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        if "embed" not in obs_specs:
            raise ValueError("Input specs must contain an 'embed' spec to pool.")
        if specs.obs["embed"].embed_dim != embed_dim:
            raise ValueError(
                f"Input embed spec has embed_dim {specs.obs['embed'].embed_dim}, but tokenizer embed_dim is {embed_dim}"
            )
        obs_specs["embed"] = new_spec
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [KeyMapping(in_keys=[("obs", "embed")], out_keys=[("obs", "embed")])]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(self, obs_embed: Tensor) -> Tensor:
        obs_embed = self.transformer(obs_embed)
        obs_embed = self.pooling(obs_embed)

        if obs_embed.is_nested:
            # TODO: we could convert back to a normal tensor if we wanted,
            # which may improve performance
            pass

        obs_embed = self.final_norm(obs_embed)
        return obs_embed

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(transformer={self.transformer}, pooling={self.pooling})"
