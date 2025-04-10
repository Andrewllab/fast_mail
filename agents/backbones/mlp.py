from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Type

import torch.nn as nn

if TYPE_CHECKING:
    from torch import Tensor


def linear_action_prediction(embed_dim: int, action_dim: int) -> nn.Module:
    return MlpModel(in_features=embed_dim, hidden_sizes=None, out_features=action_dim)


def mlp_action_prediction(
    embed_dim: int,
    action_dim: int,
    hidden_sizes: int | Sequence[int],
    non_linearity: Type[nn.Module],
) -> nn.Module:
    return MlpModel(
        in_features=embed_dim,
        out_features=action_dim,
        hidden_sizes=hidden_sizes,
        nonlinearity=non_linearity,
    )


class MlpModel(nn.Module):
    """Multilayer Perceptron with last layer linear.

    Args:
        in_features (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        out_features: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int | None,
        hidden_sizes: int | Sequence[int] | None,
        nonlinearity: type[nn.Module] | str = nn.ReLU,
    ):
        super().__init__()
        self._in_features = in_features

        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        elif hidden_sizes is None:
            hidden_sizes = []
        else:
            hidden_sizes = list(hidden_sizes)

        if isinstance(nonlinearity, str):
            nonlinearity = getattr(nn, nonlinearity)
            assert issubclass(nonlinearity, nn.Module)

        hidden_layers = [
            nn.Linear(n_in, n_out)
            for n_in, n_out in zip([in_features] + hidden_sizes[:-1], hidden_sizes)
        ]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend([layer, nonlinearity()])
        if out_features is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else in_features
            sequence.append(nn.Linear(last_size, out_features))
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
