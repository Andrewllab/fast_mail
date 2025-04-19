from __future__ import annotations

from typing import Callable, Sequence

import torch.nn as nn
from torch import Tensor


class MlpModel(nn.Module):
    """Multilayer Perceptron model that duck-types with a Linear layer.

    Args:
        in_features (int): number of input features.
        out_features (int): number of output features.
        hidden_sizes (int | Sequence[int] | None): Hidden layer sizes. If None,
            no hidden layers are used.
        activation (type[nn.Module] | str): Activation function. Default is ReLU.
        norm (type[nn.Module] | str | None): Normalization layer. Default is None.
        bias (bool): Whether to use bias in the Linear layers. Default is True.
        dropout (float | None): Dropout probability of each hidden embedding.
            If greater than 0, dropout is applied after activation layer and
            (maybe) norm layer. Default is None.
        norm_first (bool): Whether to apply normalization before activation.
            Default is True.
        plain_last (bool): If False, the non-linearity, batch normalization and
            dropout are applied to the last layer as well.

    Modified from:
        - https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/models/mlp.py
        - https://github.com/astooke/rlpyt/blob/master/rlpyt/models/mlp.py
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_sizes: int | Sequence[int] | None = None,
        activation: Callable[[], nn.Module] | str | None = nn.ReLU,
        norm: Callable[[int | tuple[int, ...]], nn.Module] | str | None = None,
        bias: bool = True,
        dropout: float | None = None,
        norm_first: bool = True,
        plain_last: bool = True,
    ):
        super().__init__()
        self._in_features = in_features

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif hidden_sizes is None:
            hidden_sizes = []
        else:
            hidden_sizes = list(hidden_sizes)

        if not plain_last:
            hidden_sizes = hidden_sizes + [out_features]

        if isinstance(activation, str):
            activation = getattr(nn, activation)
            assert issubclass(activation, nn.Module)

        if isinstance(norm, str):
            norm = getattr(nn, norm)
            assert issubclass(norm, nn.Module)

        linears = [
            nn.Linear(n_in, n_out)
            for n_in, n_out in zip([in_features] + hidden_sizes[:-1], hidden_sizes)
        ]

        sequence = list()
        for linear in linears:
            layer: list[nn.Module] = [linear]
            if activation is not None:
                layer.append(activation())

            if norm is not None:
                _norm = norm(linear.out_features)
                if norm_first:
                    layer.insert(1, _norm)
                else:
                    layer.append(_norm)

            if dropout is not None and dropout > 0:
                layer.append(nn.Dropout(dropout))

            sequence.extend(layer)

        if plain_last:
            last_size = hidden_sizes[-1] if hidden_sizes else in_features
            sequence.append(nn.Linear(last_size, out_features, bias=bias))

        self.model = nn.Sequential(*sequence)
        self._out_features = hidden_sizes[-1] if out_features is None else out_features

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)

    @property
    def in_features(self) -> int:
        """Returns the input size of the model."""
        return self._in_features

    @property
    def out_features(self) -> int:
        """Retuns the output size of the model."""
        return self._out_features

    def __repr__(self) -> str:
        return repr(self.model)
