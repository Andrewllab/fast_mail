from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Union

import tensordict
import torch
from tensordict import TensorDict
from torch.utils.data._utils.collate import collate

from utils.nested import cat_nested

if TYPE_CHECKING:
    from torch_geometric.data import Batch
    from torch_geometric.data.data import BaseData


def collate_tensor_dict(
    batch: list[TensorDict],
    *,
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
) -> TensorDict:
    # We slightly modify how the default collate function treats MutableMappings,
    # which assumes that iter(elem) yields keys. In a TensorDict, iter(elem)
    # causes a StopIteration error.
    # For reference see: https://github.com/pytorch/pytorch/blob/main/torch/utils/data/_utils/collate.py#L165
    # Note: we cannot use tensordict.stack here because it does not treat
    # non-tensor values such as pyg Data and strings correctly. The
    # collate_fn_map does this for us.

    elem = batch[0]

    # TODO: The correct shape should include the shape of the element. For now,
    # we disable this as a hack so that the obs tensordict only has one batch
    # dimension, allowing us to insert the obs embedding without shape mismatch.
    # stacked_shape = (len(batch),) + elem.shape
    stacked_shape = (len(batch),)

    # recursively collate each key into a dictionary
    d = {
        key: collate([d[key] for d in batch], collate_fn_map=collate_fn_map)
        # must call .keys() to avoid StopIteration error
        for key in elem.keys()
    }

    # When assigning a list to a key in a TensorDict, wrap it in a numpy array
    # (which is then wrapped again in a NonTensorData), instead of creating a
    # NonTensorStack. NonTensorStack causes pin_memory() to crash because it
    # is an instance of MutableMapping even though it should not be.
    with tensordict.set_list_to_stack(False):

        # We explicitly set the shape instead of using auto_batch_size_,
        # because NonTensorData inherits the batch size of the tensordict when
        # it is created, and cannot be changed after the fact.
        td = TensorDict(d, batch_size=stacked_shape, device=elem.device)

    return td


def collate_torch_geom(
    batch: list[BaseData],
    *,
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
) -> Batch:
    """Collate a list of Data or HeteroData objects into a single Batch.
    Modified from torch_geometric.loader.dataloader.Collater.__call__.
    """
    from torch_geometric.data import Batch

    return Batch.from_data_list(batch)


def collate_tensor_fn(
    batch,
    *,
    collate_fn_map: Optional[dict[Union[type, tuple[type, ...]], Callable]] = None,
):
    """A collate function that handles jagged nested tensors too
    from torch.utils.data._utils.collate import collate_tensor_fn
    """

    elem = batch[0]
    out = None
    if elem.is_nested and elem.layout == torch.jagged:
        # handle jagged nested tensors
        return cat_nested(batch, dim=0)

    if elem.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_bsr,
        torch.sparse_csc,
        torch.sparse_bsc,
    }:
        raise RuntimeError(
            "Batches of sparse tensors are not currently supported by the default collate_fn; "
            "please provide a custom collate_fn to handle them appropriately."
        )
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel = sum(x.numel() for x in batch)
        storage = elem._typed_storage()._new_shared(numel, device=elem.device)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.stack(batch, 0, out=out)


def update_collate_fn_map():
    """Add collate function for tensordict to the default collate function."""
    from torch.utils.data._utils.collate import default_collate_fn_map

    default_collate_fn_map.update(
        {
            TensorDict: collate_tensor_dict,
            torch.Tensor: collate_tensor_fn,
        }
    )

    try:
        from torch_geometric.data.data import BaseData

        default_collate_fn_map.update(
            {
                # this is used for both Data and HeteroData
                BaseData: collate_torch_geom,
            }
        )

    except ImportError:
        pass
