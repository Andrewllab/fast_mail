from typing import Callable

import torch
import torch.nn as nn


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
