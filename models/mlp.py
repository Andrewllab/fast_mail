from __future__ import annotations

from typing import Callable, Sequence

import torch.nn as nn
from torch import Tensor

from utils.nested import flatten_nested_tensor, unflatten_nested_tensor


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
        norm: Callable[[int | tuple[int, ...]], nn.Module] | str | None = None,
        activation: Callable[[], nn.Module] | str | None = nn.ReLU,
        bias: bool | Sequence[bool] = True,
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

        if not isinstance(bias, bool):
            biases = list(bias)
            if len(biases) != len(hidden_sizes) + 1:
                raise ValueError(
                    f"Length of `bias` vector ({len(biases)}) does not match number of layers {len(hidden_sizes) + 1}"
                )
        else:
            biases = [bias] * (len(hidden_sizes) + 1)

        if not plain_last:
            hidden_sizes = hidden_sizes + [out_features]

        if isinstance(activation, str):
            activation = getattr(nn, activation)
            assert issubclass(activation, nn.Module)

        if isinstance(norm, str):
            norm = getattr(nn, norm)
            assert issubclass(norm, nn.Module)

        in_sizes = [in_features] + hidden_sizes[:-1]

        linears = [
            nn.Linear(n_in, n_out, bias=b)
            for n_in, n_out, b in zip(in_sizes, hidden_sizes, biases)
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
            sequence.append(nn.Linear(last_size, out_features, bias=biases[-1]))

        self.model = nn.Sequential(*sequence)
        self._out_features = hidden_sizes[-1] if out_features is None else out_features

    def forward(self, input: Tensor) -> Tensor:
        # Handling shapes for multi-dimensional inputs or nested tensors
        leading_dims = None
        orig_offsets = None
        orig_vshape = None
        if input.ndim > 2 and input.is_nested:
            input, orig_offsets, orig_vshape = flatten_nested_tensor(
                input, start_dim=0, end_dim=1
            )
        elif input.ndim > 2:
            leading_dims = input.shape[:-1] if input.ndim > 2 else None
            input = input.flatten(start_dim=0, end_dim=-2)

        # Actual forward pass
        output = self.model(input)

        # Unhandling shapes to restore original dimensions
        if leading_dims is not None:
            output = output.unflatten(dim=0, sizes=leading_dims)
        if orig_offsets is not None and orig_vshape is not None:
            output = unflatten_nested_tensor(
                output, orig_offsets, orig_vshape, start_dim=0, end_dim=1
            )
        return output

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
