from __future__ import annotations

import logging
import re
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.typing import WITH_TORCH_CLUSTER, torch_cluster

log = logging.getLogger(__name__)


def batch2ptr(batch: Tensor, batch_size: int | None = None) -> Tensor:
    """Convert batch vector to ptr vector.
    Args:
        batch (Tensor): Batch vector of shape (N,) which assigns each point to a
            specific example in the batch.
        batch_size (int, optional): The number of examples in the batch. If not
            provided, it is inferred from the batch vector.
    Returns:
        Tensor: ptr vector of shape (batch_size + 1,) which indicates the
            boundaries of each example in the batch.

    Reference: https://github.com/rusty1s/pytorch_cluster/blob/master/torch_cluster/knn.py#L77
    """
    batch_size = batch_size if batch_size is not None else batch.max() + 1
    arange = torch.arange(batch_size + 1, device=batch.device)
    ptr = torch.bucketize(arange, batch)
    return ptr


def ptr2batch(ptr: Tensor) -> Tensor:
    """Convert ptr vector to batch vector.
    Args:
        ptr (Tensor): ptr vector of shape (batch_size + 1,) which indicates the
            boundaries of each example in the batch.
    Returns:
        Tensor: Batch vector of shape (N,) which assigns each point to a
            specific example in the batch.
    """
    lengths = ptr2lengths(ptr)
    return lengths2batch(lengths)


def lengths2ptr(lengths: Tensor) -> Tensor:
    """Convert lengths vector to ptr vector.
    Args:
        lengths (Tensor): vector of shape (batch_size,) which indicates the
            size of each batch element.
    Returns:
        Tensor: ptr vector of shape (batch_size + 1,) which indicates the
            boundaries of each example in the batch.
    """
    ptr = lengths.new_zeros(lengths.size(0) + 1)
    torch.cumsum(lengths, dim=0, out=ptr[1:])
    return ptr


def ptr2lengths(ptr: Tensor) -> Tensor:
    """Convert ptr vector to lengths vector.
    Args:
        ptr (Tensor): ptr vector of shape (batch_size + 1,) which indicates the
            boundaries of each example in the batch.
    Returns:
        Tensor: vector of shape (batch_size,) which indicates the size of each
            batch element.
    """
    return ptr[1:] - ptr[:-1]


def lengths2batch(lengths: Tensor) -> Tensor:
    """Convert lengths vector to batch vector.
    Args:
        lengths (Tensor): vector of shape (batch_size,) which indicates the
            size of each batch element.
    Returns:
        Tensor: Batch vector of shape (N,) which assigns each point to a
            specific example in the batch.
    """
    return torch.repeat_interleave(lengths)


def batch2lengths(batch: Tensor, batch_size: int | None = None) -> Tensor:
    """Convert batch vector to lengths vector.
    Args:
        batch (Tensor): Batch vector of shape (N,) which assigns each point to a
            specific example in the batch.
        batch_size (int, optional): The number of examples in the batch. If not
            provided, it is inferred from the batch vector.
    Returns:
        Tensor: vector of the size of each batch element (batch_size,).
    """
    ptr = batch2ptr(batch, batch_size=batch_size)
    return ptr2lengths(ptr)


def offset2ptr(offset: Tensor) -> Tensor:
    """Convert offset vector to batch vector.
    Args:
        offset (Tensor): offset vector of shape (batch_size,) which indicates
            the end index of each example in the batch,
    Returns:
        Tensor: ptr vector of shape (batch_size + 1,) which indicates the
            boundaries of each example in the batch.
    """
    return F.pad(offset, (1, 0), value=0).long()


def ptr2offset(ptr: Tensor) -> Tensor:
    """Convert ptr vector to offset vector.
    Args:
        ptr (Tensor): ptr vector of shape (batch_size + 1,) which indicates the
            boundaries of each example in the batch.
    Returns:
        Tensor: offset vector of shape (batch_size,) which indicates the end
            index of each example in the batch,
    """
    return ptr[1:].long()


def fps(
    x: Tensor,
    ptr: Tensor | None = None,
    batch: Tensor | None = None,
    ratio: float | None = None,
    n_points: int | None = None,
    random_start: bool = True,
) -> Tensor:
    r""" "A sampling algorithm from the `"PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
    most distant point with regard to the rest points.

    Args:
        x (Tensor): Point feature matrix of shape (N, D).
        ptr (LongTensor, optional): ptr vector of shape (batch_size + 1,) which indicates the
            boundaries of each example in the batch.
        batch (LongTensor, optional): Batch vector of shape (N,) which assigns each point to a
            specific example in the batch.
        ratio (float, optional): Ratio of points to sample. Either ratio or n_points must be set.
        n_points (int, optional): Number of points to sample per example. Either ratio or n_points
            must be set.
        random_start (bool, optional): If set to `False`, use the first point as the starting
            point. (default: `True`)

    Returns:
        Tensor: Indices of the sampled points.
    """
    if ratio is not None and n_points is not None:
        raise ValueError("Only one of ratio or n_points can be set.")

    if ptr is None:
        if batch is None:
            # create ptr vector for batch with single element
            ptr = torch.tensor([0, x.size(0)], device=x.device)
        else:
            # create ptr from batch
            ptr = batch2ptr(batch)
    elif batch is not None:
        raise ValueError("Only one of ptr or batch can be set.")

    if x.numel() == 0:
        # empty point cloud
        return torch.empty(0, dtype=torch.long, device=x.device)

    # Remove any empty pointclouds in the batch. Empty pointclouds make it very
    # likely that the first point of the next non-empty pointcloud is selected,
    # which is undesirable behavior.
    # If n_points is set, ratio must have the same number of elements as ptr - 1,
    # so we filter out the empty pointclouds before computing ratio.
    ptr = ptr.unique()

    if n_points is not None:
        # compute a unique ratio per example in the batch
        lengths = ptr2lengths(ptr)
        # to avoid rounding issues, since pyg_fps uses ceil internally
        ratio = (n_points - 0.01) / lengths.float()
        # ensure ratio does not exceed 1.0
        ratio.clamp_max_(1.0)

    elif ratio is not None:
        pass

    else:
        raise ValueError("One of ratio or n_points must be set.")

    # we call pyg_fps with ptr instead of batch to prevent it from recomputing
    # ptr internally
    idxs = torch_cluster.fps(x, ratio=ratio, random_start=random_start, ptr=ptr)

    if n_points is not None:
        # ensure that we have exactly n_points per example
        assert torch.all(
            batch2lengths(ptr2batch(ptr)[idxs])
            == torch.clamp(torch.tensor(n_points, device=x.device), max=lengths)
        )

    return idxs


def knn(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int,
    ptr_x: torch.Tensor | None = None,
    ptr_y: torch.Tensor | None = None,
    batch_x: torch.Tensor | None = None,
    batch_y: torch.Tensor | None = None,
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: int | None = None,
    pad_too_small: bool = False,
) -> torch.Tensor:
    r"""Finds for each element in :obj:`y` the :obj:`k` nearest points in
    :obj:`x`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{M \times F}`.
        k (int): The number of neighbors.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. :obj:`batch_y` needs to be sorted.
            (default: :obj:`None`)
        cosine (boolean, optional): If :obj:`True`, will use the Cosine
            distance instead of the Euclidean distance to find nearest
            neighbors. (default: :obj:`False`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
        batch_size (int, optional): The number of examples :math:`B`.
            Automatically calculated if not given. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import knn

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        y = torch.Tensor([[-1, 0], [1, 0]])
        batch_y = torch.tensor([0, 0])
        assign_index = knn(x, y, 2, batch_x, batch_y)
    """
    if not WITH_TORCH_CLUSTER:
        raise ImportError("`knn` requires `torch-cluster`")

    if x.numel() == 0 or y.numel() == 0:
        # this is also correct if padding is requested, as 0 still evenly
        # divides by k
        # essentially, `assert 0 == y.size(0) * k`
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.unsqueeze(dim=-1) if x.dim() == 1 else x
    y = y.unsqueeze(dim=-1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    if ptr_x is None or ptr_y is None:
        # determine "global" batch size across both inputs
        if batch_size is None:
            batch_size = 1

            if ptr_x is not None:
                assert x.size(0) == ptr_x[-1]
                batch_size = ptr_x.numel() - 1
            elif batch_x is not None:
                assert x.size(0) == batch_x.numel()
                batch_size = batch_x[-1].item() + 1

            if ptr_y is not None:
                assert y.size(0) == ptr_y[-1]
                batch_size = max(batch_size, ptr_y.numel() - 1)
            elif batch_y is not None:
                assert y.size(0) == batch_y.numel()
                batch_size = max(batch_size, batch_y[-1].item() + 1)

        assert batch_size is not None and batch_size > 0

        if batch_size > 1:
            # create ptr vectors from batch vectors
            if ptr_x is None:
                assert batch_x is not None
                ptr_x = batch2ptr(batch_x, batch_size=batch_size)
            if ptr_y is None:
                assert batch_y is not None
                ptr_y = batch2ptr(batch_y, batch_size=batch_size)
        else:
            # create ptr vectors for batch with single element
            ptr_x = torch.tensor([0, x.size(0)], device=x.device)
            ptr_y = torch.tensor([0, y.size(0)], device=y.device)

    assert ptr_x is not None and ptr_y is not None
    idxs = torch.ops.torch_cluster.knn(x, y, ptr_x, ptr_y, k, cosine, num_workers)

    if pad_too_small:
        # pad idxs to ensure that each point in x has at least k neighbors found
        idxs_y = idxs[0]  # idxs_y index the tensor y
        # Note: idxs_y is sorted so it has the form of a batch vector
        n_neighbors = batch2lengths(idxs_y)
        invalid = n_neighbors < k
        if torch.any(invalid):
            # find which clusters are too small
            idxs_y_invalid = torch.nonzero(invalid).squeeze(dim=-1)

            # find which batch elements these belong to and log a warning
            batch_idx_invalid = (
                torch.bucketize(idxs_y_invalid, ptr_y[1:], right=True)
            ).unique()
            lengths_invalid = ptr2lengths(ptr_x)[batch_idx_invalid]
            log.warning(
                "Padding knn matches because batch elements %s only have %s points, which is less than k=%d.",
                batch_idx_invalid.tolist(),
                lengths_invalid.tolist(),
                k,
            )

            # clamp n_neighbors to be at least k
            new_n_neighbors = n_neighbors.clamp_min(k)

            # allocate new idxs according to new n_neighbors
            new_idxs = idxs.new_zeros((idxs.size(0), new_n_neighbors.sum()))

            # compute new y indices according to new n_neighbors
            new_idxs_y = lengths2batch(new_n_neighbors)
            new_idxs[0] = new_idxs_y

            # copy over old x indices for valid clusters
            mask_new = ~torch.isin(new_idxs_y, idxs_y_invalid)
            mask_old = ~torch.isin(idxs_y, idxs_y_invalid)
            new_idxs[1, mask_new] = idxs[1, mask_old]

            # for clusters without enough neighbors, pad by repeating until
            # we have at least k neighbors, then truncate to k neighbors
            for idx in idxs_y_invalid:
                mask_new = new_idxs_y == idx
                mask_old = idxs_y == idx
                padding_factor = torch.ceil(k / n_neighbors[idx]).int()
                padded_values = idxs[1, mask_old].repeat(padding_factor)
                new_idxs[1, mask_new] = padded_values[:k]

            idxs = new_idxs

        assert idxs.size(1) == y.size(0) * k

    return idxs


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
