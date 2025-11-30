from typing import Callable

import torch
import torch.nn as nn

from utils.nested import as_nested_view, to_strided_tensor


# SwishGLU -- A Gated Linear Unit (GLU) with the Swish activation; always better than GELU MLP!
class SwishGLU(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, bias: bool = True, device=None, dtype=None
    ) -> None:
        super().__init__()
        self.act = nn.SiLU()
        self.project = nn.Linear(
            in_dim, 2 * out_dim, bias=bias, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected, gate = self.project(x).tensor_split(2, dim=-1)
        return projected * self.act(gate)


class TransformerEncoderLayer(nn.Module):
    """Explicit differences from nn.TransformerEncoderLayer:

    - No need for src_key_padding_mask with nested tensors :)
    - Only supports batch_first=True: nested tensors do not support seq_len as the
    first dimension
    - unnecessary fast path logic is removed
    """

    def __init__(
        self,
        embed_dim: int,
        norm: type[nn.Module],
        attention: type[nn.Module],
        residual_dropout: float | Callable[[], nn.Module],
        mlp1: type[nn.Module],
        activation: type[nn.Module] | str,
        mlp_dropout: float | Callable[[], nn.Module],
        mlp2: type[nn.Module],
        norm_first: bool,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = attention(embed_dim, **factory_kwargs)
        self.norm1 = norm(embed_dim, **factory_kwargs)
        self.norm2 = norm(embed_dim, **factory_kwargs)

        if isinstance(residual_dropout, float):
            self.residual_dropout = nn.Dropout(residual_dropout)
        else:
            self.residual_dropout = residual_dropout()

        if isinstance(mlp_dropout, float):
            mlp_dropout_p = mlp_dropout  # store reference to float value
            mlp_dropout = lambda: nn.Dropout(mlp_dropout_p)

        if isinstance(activation, str):
            activation = getattr(torch.nn, activation)

        self.ff_block = nn.Sequential(
            mlp1(embed_dim, 4 * embed_dim, **factory_kwargs),
            activation(),
            mlp_dropout(),
            mlp2(4 * embed_dim, embed_dim, **factory_kwargs),
            mlp_dropout(),
        )

    def _sa_block(self, x, attn_mask):
        x = self.self_attn(x, attn_mask=attn_mask)
        return self.residual_dropout(x)

    def forward(self, x, attn_mask=None):
        """
        Arguments:
            src: (batch_size, seq_len, embed_dim)
            attn_mask: (batch_size, seq_len, seq_len)
        """
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), attn_mask=attn_mask)
            x = x + self.ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_mask=attn_mask))
            x = self.norm2(x + self.ff_block(x))
        return x


class AttentionPoolingLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_tokens: int,
        norm: type[nn.Module],
        attention: type[nn.Module],
        residual_dropout: float | Callable[[], nn.Module],
        mlp1: type[nn.Module],
        activation: type[nn.Module] | str,
        mlp_dropout: float | Callable[[], nn.Module],
        mlp2: type[nn.Module],
        norm_first: bool,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = attention(output_dim, input_dim, input_dim, **factory_kwargs)
        self.norm1 = norm(input_dim if norm_first else output_dim, **factory_kwargs)
        self.norm2 = norm(output_dim, **factory_kwargs)

        if isinstance(residual_dropout, float):
            self.residual_dropout = nn.Dropout(residual_dropout)
        else:
            self.residual_dropout = residual_dropout()

        if isinstance(mlp_dropout, float):
            mlp_dropout_p = mlp_dropout  # store reference to float value
            mlp_dropout = lambda: nn.Dropout(mlp_dropout_p)

        if isinstance(activation, str):
            activation = getattr(torch.nn, activation)

        self.ff_block = nn.Sequential(
            mlp1(output_dim, 4 * output_dim, **factory_kwargs),
            activation(),
            mlp_dropout(),
            mlp2(4 * output_dim, output_dim, **factory_kwargs),
            mlp_dropout(),
        )

        self.query_tokens = nn.Parameter(
            torch.randn(n_tokens, output_dim, **factory_kwargs)
        )

    def _sa_block(self, x, queries, attn_mask):
        x = self.self_attn(queries, x, x, attn_mask=attn_mask)
        return self.residual_dropout(x)

    def forward(self, x, attn_mask=None):
        """
        Arguments:
            src: (batch_size, seq_len, input_dim)
            attn_mask: (batch_size, seq_len, seq_len)
        """
        # x: (batch, sequence_len, dim)
        # query_tokens: (num_tokens, dim) → add batch dimension
        queries = self.query_tokens.unsqueeze(0).expand(x.size(0), -1, -1)

        if x.is_nested:
            # if any of the inputs are nested, we need to convert the query tokens
            # to a nested tensor as well
            # we try to create a view, but there is no way to avoid a copy here
            # since queries is a non-contiguous view of a leaf Variable
            queries = as_nested_view(queries)

        if self.norm_first:
            x = queries + self._sa_block(self.norm1(x), queries, attn_mask=attn_mask)
            x = x + self.ff_block(self.norm2(x))
        else:
            x = self.norm1(queries + self._sa_block(x, queries, attn_mask=attn_mask))
            x = self.norm2(x + self.ff_block(x))

        if x.is_nested:
            # we know that every batch element has the same number of tokens,
            # so we can convert back to a normal tensor
            x = to_strided_tensor(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        norm: type[nn.Module],
        attention: type[nn.Module],
        residual_dropout: type[nn.Module],
        mlp1: type[nn.Module],
        activation: type[nn.Module] | str,
        mlp_dropout: type[nn.Module],
        mlp2: type[nn.Module],
        norm_first: bool,
        device=None,
        dtype=None,
    ):
        super().__init__()

        layer_factory = lambda: TransformerEncoderLayer(
            embed_dim=embed_dim,
            norm=norm,
            attention=attention,
            residual_dropout=residual_dropout,
            mlp1=mlp1,
            activation=activation,
            mlp_dropout=mlp_dropout,
            mlp2=mlp2,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )

        self.layers = nn.Sequential(*[layer_factory() for _ in range(n_layers)])
        self.norm = norm(embed_dim)

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.norm(x)
        return x
