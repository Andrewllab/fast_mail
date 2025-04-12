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
        norm_module: type[nn.Module],
        attention_module: type[nn.Module],
        residual_dropout_module: type[nn.Module],
        mlp1_module: type[nn.Module],
        activation: type[nn.Module] | str,
        mlp_dropout_module: type[nn.Module],
        mlp2_module: type[nn.Module],
        norm_first: bool,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.norm_first = norm_first

        self.self_attn = attention_module(embed_dim, **factory_kwargs)
        self.norm1 = norm_module(embed_dim, **factory_kwargs)
        self.norm2 = norm_module(embed_dim, **factory_kwargs)

        self.residual_dropout = residual_dropout_module()

        if isinstance(activation, str):
            activation = getattr(torch.nn, activation)

        self.ff_block = nn.Sequential(
            mlp1_module(embed_dim, 4 * embed_dim, **factory_kwargs),
            activation(),
            mlp_dropout_module(),
            mlp2_module(4 * embed_dim, embed_dim, **factory_kwargs),
            mlp_dropout_module(),
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
        layer_module: type[nn.Module],
        norm_module: type[nn.Module],
        embed_dim: int,
        n_layers: int,
    ):
        super().__init__()
        self.blocks = nn.Sequential(*[layer_module(embed_dim) for _ in range(n_layers)])
        self.norm = norm_module(embed_dim)

    def forward(self, x, attn_mask=None):
        for layer in self.blocks:
            x = layer(x, attn_mask=attn_mask)
        x = self.norm(x)
        return x
