from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Union

import tensordict
from tensordict import TensorDict
from torch.utils.data._utils.collate import collate

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


def update_collate_fn_map():
    """Add collate function for tensordict to the default collate function."""
    from torch.utils.data._utils.collate import default_collate_fn_map

    default_collate_fn_map.update(
        {
            TensorDict: collate_tensor_dict,
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


update_collate_fn_map()
