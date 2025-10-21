import re
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter


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
    lengths = scatter(torch.ones_like(batch.batch), batch.batch, dim_size=batch_size)
    batch.ptr = F.pad(lengths.cumsum(dim=0), (1, 0))

    data_keys = [key for key in batch.keys() if key not in ("ptr", "batch")]

    # for homogeneous data, slice_dict is the same as ptr for each field
    batch._slice_dict = {key: batch.ptr.clone() for key in data_keys}

    if update_batch_size:
        batch._num_graphs = lengths.numel()

        # for homogeneous data, inc_dict is zero for each field
        batch._inc_dict = {key: lengths.new_zeros() for key in data_keys}
    else:
        assert batch._num_graphs == lengths.numel()
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
    # recreate batch from ptr
    lengths = ptr[1:] - ptr[:-1]
    store.batch = torch.repeat_interleave(lengths)

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
