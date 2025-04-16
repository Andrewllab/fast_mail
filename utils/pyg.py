import re

import torch
from torch import Tensor
from torch_geometric.data import Data
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
    batch = data.batch

    for key, value in data.items():
        if key == "num_nodes":
            data.num_nodes = mask.sum().item()
        elif key == "ptr":
            assert batch is not None
            data.ptr = scatter(mask.to(torch.int64), batch, dim_size=data.batch_size)
        elif bool(re.search("edge", key)):
            continue
        elif (
            isinstance(value, Tensor)
            and value.size(0) == num_nodes
            and value.size(0) != 1
        ):
            data[key] = value[mask]

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
