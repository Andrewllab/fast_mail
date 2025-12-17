from typing import Sequence

import torch
from torch import Tensor

from utils.pyg import batch2ptr


def cat_nested(tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
    r"""Concatenates a sequence of nested tensors along a given dimension.
    If none of the tensors are nested, this is equivalent to :meth:`torch.cat`.
    """
    if not any(t.is_nested for t in tensors):
        # this also handles the empty list case
        return torch.cat(tensors, dim=dim)

    # convert dim to a positive integer
    dim %= tensors[0].ndim

    ragged_dim = next(t._ragged_idx for t in tensors if t.is_nested)

    # convert any non-nested tensors to nested tensors
    tensors = [
        (as_nested_view(t, ragged_dim=ragged_dim) if not t.is_nested else t)
        for t in tensors
    ]

    if dim == ragged_dim:
        dim -= 1  # jagged dim is removed from values tensor

        # split each nested tensor into its elements (view, not copy)
        values = [t.unbind(dim=dim) for t in tensors]
        # transpose the list of lists
        # [[A[0], A[1], A[2], ...], B[0], B[1], B[2], ...], ...]
        # -> [[A[0], B[0], C[0], ...], [A[1], B[1], C[1], ...], ...]
        values = list(zip(*values))
        # flatten into a single list
        values = [v for sublist in values for v in sublist]
        # concatenate all values along the specified dim
        values = torch.cat(values, dim=dim)

        # offsets sum, since each jagged element is a concatenation of the
        # corresponding jagged elements of the input tensors
        offsets = torch.stack([t.offsets() for t in tensors]).sum(dim=0)

        return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)

    elif dim < ragged_dim:
        values = torch.cat([t.values() for t in tensors], dim=dim)

        # offsets are accumulated from all tensors
        offsets = []
        offset = 0
        for t in tensors:
            t_offsets = t.offsets()
            # shift offsets by the current offset
            offsets.append(t_offsets[:-1] + offset)
            offset += t_offsets[-1:]
        offsets.append(offset)
        offsets = torch.cat(offsets)

        return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


def as_nested_view(tensor: Tensor, ragged_dim: int = 1) -> Tensor:
    r"""Returns a nested tensor view of the input strided tensor without
    copying data.
    """
    # convert dim to a positive integer
    ragged_dim %= tensor.ndim

    if ragged_dim < 1:
        raise ValueError("ragged_dim must be at least 1")

    B, L = tensor.shape[ragged_dim - 1 : ragged_dim + 1]
    offsets = torch.arange(0, (B + 1) * L, step=L, device=tensor.device)
    tensor = tensor.flatten(start_dim=ragged_dim - 1, end_dim=ragged_dim)
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
        offsets = batch2ptr(batch, batch_size=batch_size)
    else:
        raise ValueError("Either batch or ptr must be provided.")

    return torch.nested.nested_tensor_from_jagged(values, offsets=offsets)


def make_jagged_nested_tensors_compatible(
    a: Tensor,
    b: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Make two jagged NestedTensors compatible for pointwise binary ops by ensuring
    they reference the *same offsets Tensor object*.

    This only works if the ragged structure is already identical (offsets equal).
    It does NOT pad / change lengths.

    Args:
        a, b: torch.NestedTensor with layout=torch.jagged

    Returns:
        b_compat
    """
    # --- Offsets checks ---
    a_off = a.offsets()
    b_off = b.offsets()

    # 1) Same *values* (ragged structure identical)
    if not torch.equal(a_off, b_off):
        raise RuntimeError(
            "offsets differ by value; ragged structures are not identical, "
            "so you can't make them compatible without padding/rebuilding."
        )

    # 2) If already the exact same Tensor object, we're done
    if a_off is b_off:
        return a, b

    # --- Rewrap to share the same offsets object (no padding) ---
    b2 = torch.nested.nested_tensor_from_jagged(values=b.values(), offsets=a_off)
    return a, b2
