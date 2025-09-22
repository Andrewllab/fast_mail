import re
from typing import TypeVar

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

    data = update_ptr(data)

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


T = TypeVar("T", bound=Data)


def update_ptr(data: T, keep_batch_size: bool = True) -> T:
    """Recompute the ptr attribute of a Batch object. While the batch attribute
    is kept up to date by transforms, the ptr attribute is usually not.

    Args:
        data (Data): The Data object.

    Returns:
        Data: The Data object with updated ptr.
    """
    if hasattr(data, "batch"):
        assert isinstance(data, Batch)
        batch = data.batch
        assert batch is not None

        batch_size = data.batch_size if keep_batch_size else None
        lengths = scatter(torch.ones_like(batch), batch, dim_size=batch_size)
        data.ptr = F.pad(lengths.cumsum(dim=0), (1, 0))

        if not keep_batch_size:
            data._num_graphs = lengths.numel()

    return data


def update_batch_metadata(data: Batch) -> Batch:
    """Recompute the ptr, _slice_dict, and _inc_dict attributes of a Batch object.
    These may be incorrect after reducing the number of nodes in some graphs,
    but need to be correct to reconstruct the batch elements from the batch.

    Args:
        data (Data): The Batch object.

    Returns:
        Data: The Batch object with updated metadata.
    """
    assert isinstance(data, Batch)

    data = update_ptr(data)

    # for homogenous graphs, the slice dict is the same as the ptr for each
    # field of the data
    for key in data._slice_dict:
        data._slice_dict[key] = data.ptr.clone()

    return data
