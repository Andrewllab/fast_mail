import math
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import h5py
import numpy as np
import torch
from tensordict import NonTensorData, TensorDict
from torch import Tensor

from utils.pyg import batch2ptr


def is_torch_nested_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor) and bool(getattr(x, "is_nested", False))


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
    if not x.is_nested:
        return x
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
    if not a.is_nested or not b.is_nested:
        return a, b
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


def _norm_dims(ndim: int, start_dim: int, end_dim: int) -> tuple[int, int]:
    start_dim %= ndim
    end_dim %= ndim
    if start_dim > end_dim:
        raise ValueError(f"start_dim ({start_dim}) must be <= end_dim ({end_dim})")
    return start_dim, end_dim


def flatten_nested_tensor(
    x: Tensor,
    start_dim: int = 0,
    end_dim: int = -1,
) -> tuple[Tensor, Optional[Tensor], Optional[tuple[int, ...]]]:
    """
    Flatten a nested tensor, returning (flattened, orig_offsets, orig_vshape).

    orig_offsets/orig_vshape are None when a plain .flatten() was sufficient.
    """
    if not x.is_nested:
        raise ValueError("Input tensor must be a nested tensor.")

    start_dim, end_dim = _norm_dims(x.ndim, start_dim, end_dim)
    ragged_dim = x._ragged_idx  # private, but kept since you depend on it
    if ragged_dim != 1:
        raise NotImplementedError("Only ragged_dim == 1 is supported.")

    # If flatten range does NOT cross ragged_dim, regular flatten is safe.
    crosses_ragged = start_dim <= ragged_dim <= end_dim
    if not crosses_ragged:
        return x.flatten(start_dim=start_dim, end_dim=end_dim), None, None

    # Case A: flatten [0..ragged_dim] -> drop nesting (return values) + metadata
    if start_dim == 0 and end_dim == ragged_dim:
        return x.values(), x.offsets(), tuple(x.shape[start_dim:end_dim])

    # Case B: flatten starting at ragged_dim and extending into value dims
    if start_dim == ragged_dim and end_dim > ragged_dim:
        orig_offsets = x.offsets()
        orig_vshape = tuple(
            x.shape[start_dim + 1 : end_dim + 1]
        )  # value dims being merged
        mult = math.prod(orig_vshape) if orig_vshape else 1

        values = x.values().view(-1, *x.shape[end_dim + 1 :])
        offsets = orig_offsets * mult

        return (
            torch.nested.nested_tensor_from_jagged(values=values, offsets=offsets),
            orig_offsets,
            orig_vshape,
        )

    raise NotImplementedError(
        "Only flattening [0..ragged_dim] or [ragged_dim..k] (k>ragged_dim) is supported."
    )


def unflatten_nested_tensor(
    x: Tensor,
    orig_offsets: Optional[Tensor] = None,
    orig_vshape: Optional[tuple[int, ...]] = None,
    start_dim: int = 0,
    end_dim: int = -1,
) -> Tensor:
    """
    Inverse of flatten_nested_tensor for the supported cases.
    If orig_offsets/orig_vshape are None, this is a no-op.
    """
    start_dim, end_dim = _norm_dims(x.ndim, start_dim, end_dim)

    # If we flattened [0..ragged_dim], we returned a plain (non-nested) values tensor.
    # So unflatten should reconstruct a nested tensor from (values=x, offsets=orig_offsets).
    if start_dim == 0:
        return torch.nested.nested_tensor_from_jagged(values=x, offsets=orig_offsets)

    # Otherwise, we flattened starting at ragged_dim and kept nesting.
    if not x.is_nested:
        raise ValueError(
            "Expected a nested tensor input when unflattening from ragged_dim."
        )

    ragged_dim = x._ragged_idx
    if ragged_dim != 1:
        raise NotImplementedError("Only ragged_dim == 1 is supported.")

    if start_dim == ragged_dim and end_dim > ragged_dim:
        # Calculate new offsets by dividing original offsets
        if orig_offsets is None:
            prod = math.prod(orig_vshape) if orig_vshape else 1
            if not torch.all(x.offsets() % prod == 0):
                raise ValueError(
                    "Cannot unflatten: current offsets are not divisible by "
                    "the product of orig_vshape dimensions."
                )
            orig_offsets = x.offsets() // prod

        values = x.values().view(
            -1, *orig_vshape, *x.shape[start_dim + len(orig_vshape) :]
        )

        return torch.nested.nested_tensor_from_jagged(
            values=values, offsets=orig_offsets
        )

    raise NotImplementedError(
        "Only unflattening [0..ragged_dim] or [ragged_dim..k] (k>ragged_dim) is supported."
    )


# -----------------------------
# Storage codec for NestedTensor (jagged)
# -----------------------------

# A marker key that is unlikely to collide with real data keys.
# (If you want even safer, prefix with your project name.)
_NT_MARKER_KEY = "__nested_jagged__"

# Field names inside the packed representation:
_NT_VALUES_KEY = "values"
_NT_OFFSETS_KEY = "offsets"
_NT_RAGGED_DIM_KEY = "ragged_dim"


def is_packed_jagged_nested_td(x: Any) -> bool:
    return (
        isinstance(x, (TensorDict, Dict, h5py.Group))
        and _NT_MARKER_KEY in x.keys()
        and _NT_VALUES_KEY in x.keys()
        and _NT_OFFSETS_KEY in x.keys()
    )


def get_packed_jagged_length(packed: TensorDict) -> int:
    """
    Get the length (number of jagged elements) of a packed jagged NestedTensor
    representation.
    """
    if not is_packed_jagged_nested_td(packed):
        raise ValueError(
            "get_packed_jagged_length expects a packed NestedTensor TensorDict"
        )

    offsets: Tensor = packed[_NT_OFFSETS_KEY]
    return int(offsets.shape[0] - 1)


def pack_jagged_nested_tensor(x: Tensor) -> TensorDict:
    """
    Convert a torch jagged NestedTensor into a pure-tensor TensorDict
    representation that is backend-saveable (memmap/hdf5/etc).

    Requirements:
      - x.is_nested == True
      - jagged layout (torch.jagged)
      - currently only supports ragged_dim == 1 (matches your utils assumptions)

    Returns:
      TensorDict with batch_size=[] containing:
        __nested_jagged__: uint8 scalar marker
        values: Tensor (x.values())
        offsets: Tensor (x.offsets())
        ragged_dim: int64 scalar
        shape: int64 1D tensor storing x.shape
    """
    if not x.is_nested:
        raise ValueError("pack_jagged_nested_tensor expects a NestedTensor")

    # You rely on _ragged_idx in several utils already; keep consistent.
    ragged_dim = int(x._ragged_idx)  # type: ignore[attr-defined]
    if ragged_dim != 1:
        raise NotImplementedError(
            "Only jagged NestedTensors with ragged_dim == 1 are supported."
        )

    # For jagged nested tensors, these are tensors and safe to store.
    values = x.values()
    offsets = x.offsets()

    packed = TensorDict(
        {
            _NT_MARKER_KEY: torch.tensor(1, dtype=torch.uint8, device=values.device),
            _NT_VALUES_KEY: values,
            _NT_OFFSETS_KEY: offsets,
            _NT_RAGGED_DIM_KEY: torch.tensor(
                ragged_dim, dtype=torch.int64, device=values.device
            ),
        },
        batch_size=[],
    )
    return packed


def unpack_jagged_nested_tensor(packed: TensorDict) -> Tensor:
    """
    Inverse of pack_jagged_nested_tensor.
    Reconstructs a torch jagged NestedTensor *view* from stored values+offsets.
    """
    if not is_packed_jagged_nested_td(packed):
        raise ValueError(
            "unpack_jagged_nested_tensor expects a packed NestedTensor TensorDict"
        )

    ragged_dim = int(packed[_NT_RAGGED_DIM_KEY].item())
    if ragged_dim != 1:
        raise NotImplementedError(
            "Only jagged NestedTensors with ragged_dim == 1 are supported."
        )

    values: Tensor = packed[_NT_VALUES_KEY]
    offsets: Tensor = packed[_NT_OFFSETS_KEY]

    # This should not copy values; it will build a NestedTensor referencing them.
    if not isinstance(values, Tensor):
        values = torch.as_tensor(values)
    if not isinstance(offsets, Tensor):
        offsets = torch.as_tensor(offsets)
    x = torch.nested.nested_tensor_from_jagged(values=values, offsets=offsets)
    return x


# -----------------------------
# Recursive pack/unpack over arbitrary nested structures
# -----------------------------


def pack_nested_for_storage(obj: Any) -> Any:
    """
    Recursively convert any jagged NestedTensor into a backend-friendly packed form.

    Supports:
      - Tensor
      - TensorDict
      - dict-like mappings
      - list/tuple

    Returns a structure with the same shape, but NestedTensors replaced by packed TensorDict.
    """
    if isinstance(obj, Tensor) and obj.is_nested:
        return pack_jagged_nested_tensor(obj)

    if isinstance(obj, TensorDict):
        # Important: ensure this container does NOT enforce a parent batch_size prefix
        # when we insert packed nested tensors with batch_size=[].
        out = obj.clone(recurse=False)
        out.auto_batch_size_(batch_dims=0)
        for k in out.keys():
            out[k] = pack_nested_for_storage(out[k])
        return out

    if isinstance(obj, MutableMapping):
        for k, v in list(obj.items()):
            obj[k] = pack_nested_for_storage(v)
        return obj

    if isinstance(obj, tuple):
        return tuple(pack_nested_for_storage(v) for v in obj)

    if isinstance(obj, list):
        return [pack_nested_for_storage(v) for v in obj]

    return obj


def unpack_nested_from_storage(obj: Any) -> Any:
    """
    Recursively convert packed nested-tensor representations back into real NestedTensors.
    """
    if is_packed_jagged_nested_td(obj):
        return unpack_jagged_nested_tensor(obj)

    if isinstance(obj, TensorDict):
        out = obj.clone(recurse=False)
        # keep batch_dims=0 to avoid prefix constraints during in-place replacement
        out.auto_batch_size_(batch_dims=0)
        for k in out.keys():
            out[k] = unpack_nested_from_storage(out[k])
        return out

    if isinstance(obj, MutableMapping):
        for k, v in list(obj.items()):
            obj[k] = unpack_nested_from_storage(v)
        return obj

    if isinstance(obj, tuple):
        return tuple(unpack_nested_from_storage(v) for v in obj)

    if isinstance(obj, list):
        return [unpack_nested_from_storage(v) for v in obj]

    return obj


def nested_unsqueeze(
    x: Tensor,
    dim: int,
) -> Tensor:
    """Unsqueeze a nested tensor along a given dimension.

    Args:
        x (Tensor): Input nested tensor.
        dim (int): Dimension to unsqueeze.

    Returns:
        Tensor: Unsqueezed nested tensor.
    """
    if not x.is_nested:
        return x.unsqueeze(dim)

    # Convert dim to positive integer
    dim %= x.ndim

    ragged_dim = x._ragged_idx  # private, but kept since you depend on it
    if dim == ragged_dim:
        raise ValueError("Cannot unsqueeze along the jagged dimension.")

    new_ragged_dim = ragged_dim + 1 if dim < ragged_dim else ragged_dim

    values = x.values().unsqueeze(dim if dim < ragged_dim else dim - 1)
    return torch.nested.nested_tensor_from_jagged(
        values, offsets=x.offsets(), jagged_dim=new_ragged_dim
    )


def nested_safe_tensordict_unsqueeze(
    td: TensorDict,
    dim: int,
    top_level: bool = True,
) -> TensorDict:
    """Unsqueeze all tensors in a tensordict, including nested tensors.

    Args:
        td (TensorDict): Input tensordict.
        dim (int): Dimension to unsqueeze.

    Returns:
        TensorDict: Unsqueezed tensordict.
    """
    out_td = TensorDict({}, batch_size=td.batch_size)
    for key, value in td.items():
        if isinstance(value, Tensor) and value.is_nested:
            out_td[key] = nested_unsqueeze(value, dim)
        elif isinstance(value, (Tensor, NonTensorData)):
            out_td[key] = value.unsqueeze(dim)
        elif isinstance(value, TensorDict):
            out_td[key] = nested_safe_tensordict_unsqueeze(value, dim, top_level=False)
            out_td[key].auto_batch_size_(batch_dims=1)
        else:
            out_td[key] = value
    if top_level:
        out_td.auto_batch_size_(batch_dims=1)
    return out_td


def nested_index(index_nt: torch.Tensor, src_nt: torch.Tensor) -> torch.Tensor:
    return torch.nested.nested_tensor(
        [emb[pid] for pid, emb in zip(index_nt.unbind(), src_nt.unbind())],
        layout=torch.jagged,
    )


def nested_index_fast(index_nt: torch.Tensor, src_nt: torch.Tensor) -> torch.Tensor:
    """
    Vectorized nested 'src_row[pid]' for jagged NestedTensors (layout=torch.jagged),
    with no Python loops.

    index_nt: jagged NT of indices, logical shape (R, jL_out) or similar
    src_nt:   jagged NT of values,  logical shape (R, jL_src, D...)
    Returns:  jagged NT, offsets from index_nt, values picked from src_nt
    """
    if not (index_nt.is_nested and src_nt.is_nested):
        raise ValueError("Expected nested tensors")

    # Offsets define the ragged rows for each NT
    idx_off = index_nt.offsets()  # (R+1,)
    src_off = src_nt.offsets()  # (R+1,)

    if idx_off.numel() != src_off.numel():
        raise RuntimeError("index_nt and src_nt must have same number of ragged rows")

    # Row id for each element in index_nt.values()
    idx_lengths = idx_off[1:] - idx_off[:-1]  # (R,)
    row_ids = torch.repeat_interleave(
        torch.arange(idx_lengths.numel(), device=idx_off.device), idx_lengths
    )  # (sum(idx_lengths),)

    # Base offset into src.values() for the corresponding row
    base = src_off[row_ids]  # (sum_idx,)

    # Flatten indices into src.values() indexing space
    pid = index_nt.values().to(base.dtype)  # ensure integer dtype compatible
    flat_idx = base + pid  # (sum_idx,)

    # Optional: bounds check (debug only; remove for speed)
    # src_lengths = (src_off[1:] - src_off[:-1])[row_ids]
    # if torch.any(pid < 0) or torch.any(pid >= src_lengths):
    #     raise RuntimeError("Out-of-bounds pid for some ragged row")

    out_vals = src_nt.values().index_select(0, flat_idx)

    # Rewrap using *index_nt's offsets* (because output has index_nt's ragged lengths)
    return torch.nested.nested_tensor_from_jagged(values=out_vals, offsets=idx_off)


def nested_index_reduced_batch(
    batch: Mapping[str, Tensor], idx: int | slice
) -> np.ndarray:
    # all other items should be actual data, e.g. position or color
    offs = batch["offsets"]
    values = batch["values"]

    if isinstance(idx, slice):
        if not (
            idx.start is not None
            and idx.stop is not None
            and idx.step is None
            and idx.stop - idx.start == 1
        ):
            raise NotImplementedError("Only slicing a single element is supported.")
        i = idx.start
    else:
        i = idx

    start = offs[i]
    end = offs[i + 1]

    values_slice = torch.from_numpy(values[start:end])  # remove batch dimension
    values_nested = torch.nested.nested_tensor_from_jagged(
        values_slice, offsets=torch.tensor([0, end - start])
    )

    return values_nested
