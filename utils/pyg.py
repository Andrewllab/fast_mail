import re
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.typing import torch_cluster


def batch2lengths(batch: Tensor, batch_size: int | None = None) -> Tensor:
    """Convert batch vector to lengths vector.
    Args:
        batch (Tensor): Batch vector of shape (N,) which assigns each point to a
            specific example in the batch.
        batch_size (int, optional): The number of examples in the batch. If not
            provided, it is inferred from the batch vector.
    Returns:
        Tensor: vector of the size of each batch element (batch_size,).

    Reference: https://github.com/rusty1s/pytorch_cluster/blob/master/torch_cluster/fps.py#L99
    """
    batch_size = batch_size if batch_size is not None else batch.max().item() + 1
    return batch.new_zeros(batch_size).scatter_add_(0, batch, torch.ones_like(batch))


def batch2ptr(batch: Tensor, batch_size: int | None = None) -> Tensor:
    """Convert batch vector to ptr vector.
    Args:
        batch (Tensor): Batch vector of shape (N,) which assigns each point to a
            specific example in the batch.
        batch_size (int, optional): The number of examples in the batch. If not
            provided, it is inferred from the batch vector.
    Returns:
        Tensor: ptr vector of shape (batch_size + 1,) which indicates the start
            index of each example in the batch, plus a final entry of N.

    Reference: https://github.com/rusty1s/pytorch_cluster/blob/master/torch_cluster/fps.py#L102
    """
    lengths = batch2lengths(batch, batch_size)
    ptr = batch.new_zeros(lengths.size(0) + 1)
    torch.cumsum(lengths, dim=0, out=ptr[1:])
    return ptr


def ptr2lengths(ptr: Tensor) -> Tensor:
    """Convert ptr vector to lengths vector.
    Args:
        ptr (Tensor): ptr vector of shape (batch_size + 1,) which indicates the start
            index of each example in the batch, plus a final entry of N.
    Returns:
        Tensor: vector of the size of each batch element (batch_size,).
    """
    return ptr[1:] - ptr[:-1]


def ptr2batch(ptr: Tensor) -> Tensor:
    """Convert ptr vector to batch vector.
    Args:
        ptr (Tensor): ptr vector of shape (batch_size + 1,) which indicates the start
            index of each example in the batch, plus a final entry of N.
    Returns:
        Tensor: Batch vector of shape (N,) which assigns each point to a
            specific example in the batch.
    """
    lengths = ptr2lengths(ptr)
    return torch.repeat_interleave(lengths)


def offset2batch(offset: Tensor) -> Tensor:
    """Convert offset vector to batch vector.
    Args:
        offset (Tensor): offset vector of shape (batch_size + 1,) which indicates the end
            index of each example in the batch,
    Returns:
        Tensor: Batch vector of shape (N,) which assigns each point to a
            specific example in the batch.
    """
    ptr = F.pad(offset, (1, 0), value=0).long()
    return ptr2batch(ptr)


def fps(
    x: Tensor,
    ptr: Tensor | None = None,
    ratio: float | None = None,
    n_points: int | None = None,
    random_start: bool = True,
) -> Tensor:
    """Farthest point sampling (FPS) for point clouds in a batch.

    Args:
        x (Tensor): Point cloud coordinates of shape (N, D).
        batch (Tensor): Batch vector of shape (N,) which assigns each point to a
            specific example in the batch.
        ratio (float): Ratio of points to sample.

    Returns:
        Tensor: Indices of the sampled points.
    """
    if ratio is not None and n_points is not None:
        raise ValueError("Only one of ratio or n_points can be set.")

    if ptr is None:
        # create ptr vector for batch with single element
        ptr = torch.tensor([0, x.size(0)], device=x.device)

    if n_points is not None:
        # compute a unique ratio per example in the batch
        lengths = ptr[1:] - ptr[:-1]
        # to avoid rounding issues, since pyg_fps uses ceil internally
        ratio = (n_points - 0.01) / lengths
        # ensure ratio does not exceed 1.0
        ratio.clamp_(max=1.0)

    elif ratio is not None:
        pass

    else:
        raise ValueError("One of ratio or n_points must be set.")

    # we call pyg_fps with ptr instead of batch to prevent it from recomputing
    # ptr internally
    indices = torch_cluster.fps(x, ratio=ratio, random_start=random_start, ptr=ptr)

    if n_points is not None:
        # ensure that we have exactly n_points per example
        assert torch.all(
            batch2lengths(ptr2batch(ptr)[indices])
            == torch.clamp(torch.tensor(n_points, device=x.device), max=lengths)
        )

    return indices


def apply_mask(data: Data, mask: Tensor) -> Data:
    """
    Apply a mask to the data or batch of data.

    Args:
        data (Data): The Data or Batch object to apply the mask to.
        mask (Tensor): A boolean Tensor, where True indicates the elements to
            keep.

    Returns:
        Data: The masked data object.

    Modified from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/fixed_points.html
    """

    num_nodes = data.num_nodes
    assert num_nodes is not None

    for key, value in data.items():
        if key == "num_nodes":
            data.num_nodes = mask.sum().item()
        elif bool(re.search("edge", key)):
            continue
        elif (
            isinstance(value, Tensor)
            and value.size(0) == num_nodes
            and value.size(0) != 1
        ):
            data[key] = value[mask]

    if isinstance(data, Batch):
        data = update_batch_metadata(data)

    return data


def apply_index(data: Data, index: Tensor) -> Data:
    """
    Apply indices to the data or batch of data.

    Args:
        data (Data): The Data or Batch object to apply the index to.
        index (Tensor): A Tensor of indices to select elements from the data.

    Returns:
        Data: The indexed data object.

    Modified from: https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/transforms/fixed_points.html
    """

    num_nodes = data.num_nodes
    assert num_nodes is not None

    for key, value in data.items():
        if key == "num_nodes":
            data.num_nodes = index.size(0)
        elif bool(re.search("edge", key)):
            continue
        elif (
            isinstance(value, Tensor)
            and value.size(0) == num_nodes
            and value.size(0) != 1
        ):
            data[key] = value[index]

    return data


def update_batch_metadata(batch: Batch, update_batch_size: bool = False) -> Batch:
    """Recompute the ptr, _slice_dict, and _inc_dict attributes of a Batch object.
    While the batch attribute is kept up to date by transforms, the ptr attribute
    is usually not. These attributes are required to reconstruct the Batch object
    correctly.

    Args:
        batch (Batch): The Batch object.

    Returns:
        Batch: The Batch object with updated metadata.
    """
    assert isinstance(batch, Data)
    assert isinstance(batch, Batch)
    assert batch.batch is not None

    batch_size = None if update_batch_size else batch.batch_size
    ptr = batch.ptr = batch2ptr(batch.batch, batch_size=batch_size)

    data_keys = [key for key in batch.keys() if key not in ("ptr", "batch")]

    # for homogeneous data, slice_dict is the same as ptr for each field
    batch._slice_dict = {key: batch.ptr.clone() for key in data_keys}

    if update_batch_size:
        batch._num_graphs = ptr.numel() - 1

        # for homogeneous data, inc_dict is zero for each field
        batch._inc_dict = {key: ptr.new_zeros(ptr.size(0) - 1) for key in data_keys}
    else:
        assert batch._num_graphs == ptr.numel() - 1
        for key in data_keys:
            if key in batch._inc_dict:
                assert torch.all(batch._inc_dict[key] == 0)
                assert batch._inc_dict[key].shape == (ptr.size(0) - 1,)
            else:
                # a new key-value pair was added
                batch._inc_dict[key] = ptr.new_zeros(ptr.size(0) - 1)
        assert all(torch.all(batch._inc_dict[key] == 0) for key in data_keys)

    return batch


def reduce_batch(batch: Batch) -> dict[str, Tensor]:
    """Reduce a Batch object to a dictionary of tensors to prepare for saving.
    The batch attribute is removed, and the ptr attribute is kept. The
    _slice_dict and _inc_dict attributes are also removed, as they can be
    reconstructed from the ptr attribute.

    Args:
        batch (Batch): The Batch object to reduce.
    Returns:
        dict: A dictionary containing the data of the Batch object.
    """
    assert isinstance(batch, Batch)

    batch = batch.to_dict()
    batch.pop("batch")  # we can reconstruct batch from ptr and save space
    return batch


def unreduce_batch(batch: Mapping[str, Tensor]) -> Batch:
    # this should have been removed in reduce_batch
    assert "batch" not in batch

    # all other items should be actual data, e.g. position or color
    ptr = batch["ptr"]

    # construct Batch object that dynamically inherits from Data (not HeteroData)
    out = Batch(_base_cls=Data)

    # batch of homogeneous data only has one store
    store = out._store

    data_keys = [key for key in batch.keys() if key != "ptr"]

    # assign all actual data to store
    for key in data_keys:
        store[key] = batch[key]

    store.ptr = ptr
    store.batch = ptr2batch(ptr)  # recreate batch from ptr

    out._num_graphs = ptr.numel() - 1
    # for homogeneous data, slice_dict is the same as ptr for each field
    out._slice_dict = {key: ptr.clone() for key in data_keys}
    # for homogeneous data, inc_dict is zero for each field
    out._inc_dict = {key: ptr.new_zeros(ptr.numel() - 1) for key in data_keys}

    return out


def index_reduced_batch(batch: Mapping[str, Tensor], idx: int | slice) -> Data:
    # this should have been removed in reduce_batch
    assert "batch" not in batch

    # all other items should be actual data, e.g. position or color
    ptr = batch["ptr"]

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

    start = ptr[i]
    end = ptr[i + 1]

    out = Data(
        **{key: value[start:end] for key, value in batch.items() if key != "ptr"}
    )

    return out
