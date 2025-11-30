from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.utils import scatter


def cat_nested(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    r"""Concatenates a sequence of nested tensors along a given dimension.
    If none of the tensors are nested, this is equivalent to :meth:`torch.cat`.
    """
    # TODO: can this be done more efficiently by manipulating offsets and values?

    if not any(t.is_nested for t in tensors):
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


def as_nested_view(tensor: Tensor) -> Tensor:
    r"""Returns a nested tensor view of the input strided tensor without
    copying data.
    """
    B, L = tensor.shape[:2]
    offsets = torch.arange(0, (B + 1) * L, step=L, device=tensor.device)
    tensor = tensor.flatten(0, 1)
    return torch.nested.nested_tensor_from_jagged(tensor, offsets=offsets)


def to_strided_tensor(
    x: Tensor,
    padding: float | None = None,
    output_size: tuple[int, ...] | None = None,
    out: Tensor | None = None,
) -> Tensor:
    r"""Converts a nested tensor to a strided ("dense") tensor. If all elements
    in the jagged dimension have the same length, this is done without copying
    by reshaping the values tensor. Otherwise, the nested tensor is converted
    by padding the elements to the same length.
    """
    offsets, values = x.offsets(), x.values()
    lengths = offsets[1:] - offsets[:-1]

    if torch.all(lengths == lengths[0]):
        # all sequences have the same length, return a reshaped tensor
        B = x.size(0)
        L = lengths[0]
        return values.unflatten(0, (B, L))

    else:
        if padding is None:
            raise ValueError(
                "padding value must be provided when padding a nested tensor to a dense tensor"
            )
        return torch.nested.to_padded_tensor(
            x, padding=padding, output_size=output_size, out=out
        )


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
        offsets = F.pad(lengths.cumsum(dim=0), (1, 0))

    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)
