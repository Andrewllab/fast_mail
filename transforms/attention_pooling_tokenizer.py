from __future__ import annotations

import logging
from functools import partial
from typing import Callable, Mapping

import torch.nn as nn
from torch import Tensor

from environments.specs import DataSpecs, EmbedSpec
from models.attention import MultiHeadAttention, MultiHeadSelfAttention
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
        attention: Mapping | Callable[..., nn.Module],
        residual_dropout: float | Callable[[], nn.Module],
        mlp1: type[nn.Module],
        activation: type[nn.Module] | str,
        mlp_dropout: float | Callable[[], nn.Module],
        mlp2: type[nn.Module],
        norm_first: bool,
        token_pos_encoder: Callable[[int, int], nn.Module] | None = None,
        input_key: str = "embed",
        output_key: str = "embed",
    ):
        super().__init__()

        if input_key not in specs.obs:
            raise ValueError(f"Key {input_key} not found in specs.obs")
        input_dim = specs.obs[input_key].embed_dim

        self.input_key = input_key
        self.output_key = output_key

        if isinstance(attention, Mapping):
            attention_kwargs = {**attention}
        elif isinstance(attention, partial):
            attention_kwargs = (
                attention.keywords if attention.keywords is not None else {}
            )
        else:
            raise ValueError(
                "attention must be a Mapping or a functools.partial instance"
            )

        self_attention = partial(MultiHeadSelfAttention, **attention_kwargs)
        cross_attention = partial(MultiHeadAttention, **attention_kwargs)

        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            n_layers=n_layers,
            norm=norm,
            attention=self_attention,
            residual_dropout=residual_dropout,
            mlp1=mlp1,
            activation=activation,
            mlp_dropout=mlp_dropout,
            mlp2=mlp2,
            norm_first=norm_first,
        )

        self.pooling = AttentionPoolingLayer(
            input_dim=input_dim,
            output_dim=embed_dim,
            n_tokens=n_tokens,
            norm=norm,
            attention=cross_attention,
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
        obs_specs[output_key] = new_spec
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=[("obs", self.input_key)], out_keys=[("obs", self.output_key)]
            )
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(self, obs_embed: Tensor) -> Tensor:
        obs_embed = self.transformer(obs_embed)
        obs_embed = self.pooling(obs_embed)
        obs_embed = self.final_norm(obs_embed)
        return obs_embed

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(transformer={self.transformer}, pooling={self.pooling})"
