from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter


def cat_nested(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    nesteds = [t.is_nested for t in tensors]

    if not any(nesteds):
        return torch.cat(tensors, dim=dim)

    # convert dim to a positive integer
    dim %= len(tensors[0].shape)

    # unbind the nested tensors and loop over each element, concatenating with
    # the corresponding elements in the other tensors
    elems = [
        torch.cat(t_elems, dim=dim - 1)
        for t_elems in zip(*(t.unbind() for t in tensors))
    ]

    return torch.nested.as_nested_tensor(elems, layout=torch.jagged)


def pyg_to_nested_tensor(
    values: Tensor,
    batch: Tensor | None = None,
    ptr: Tensor | None = None,
    batch_size: int | None = None,
) -> Tensor:
    r"""Given a contiguous batch of tensors
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`
    (with :math:`N_i` indicating the number of elements in example :math:`i`),
    creates a `nested PyTorch tensor
    <https://pytorch.org/docs/stable/nested.html>`__.
    Reverse operation of :meth:`from_nested_tensor`.

    Args:
        x (torch.Tensor): The input tensor
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times *}`.
        batch (torch.Tensor, optional): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            element to a specific example. Must be ordered.
            (default: :obj:`None`)
        ptr (torch.Tensor, optional): Alternative representation of
            :obj:`batch` in compressed format. (default: :obj:`None`)
        batch_size (int, optional): The batch size :math:`B`.
            (default: :obj:`None`)
    """
    if ptr is not None:
        offsets = ptr
    elif batch is not None:
        lengths = scatter(torch.ones_like(batch), batch, dim_size=batch_size)
        offsets = F.pad(lengths.cumsum(0), (1, 0))

    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)
