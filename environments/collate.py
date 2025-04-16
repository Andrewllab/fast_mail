from typing import Sequence

import tensordict
from tensordict import TensorDict
from torch_geometric.data import Batch as GeomBatch
from torch_geometric.data import Data as GeomData
from torch_geometric.data import HeteroData as GeomHeteroData
from torch_geometric.data.data import BaseData as GeomBaseData


def collate_tensor_dict(batch: list[TensorDict], *, collate_fn_map) -> TensorDict:
    """Collate a list of TensorDicts into a single TensorDict."""
    # tensordict.lazy_stack always produces a NonTensorStack when stacking
    # NonTensorData, and may be more performant
    # TODO: is this really better than torch.stack?
    stacked: TensorDict = tensordict.stack(batch, dim=0)

    # recompute batch size with only a single batch dimension, since tensordict
    # eagerly increases the number of batch dims when stacking
    stacked = stacked.auto_batch_size_(batch_dims=1)

    for key, value in stacked.items(include_nested=True, leaves_only=True):
        # when NonTensorStack is accessed, it returns its contents in a list
        # TODO: maybe call default_collate here instead?
        if isinstance(value, list) and isinstance(value[0], GeomData):
            torch_geom_batch = collate_torch_geom(value)
            stacked[key] = torch_geom_batch

    return stacked


def collate_torch_geom(
    batch: Sequence[GeomData] | Sequence[GeomHeteroData],
    *,
    follow_batch: list[str] | None = None,
    exclude_keys: list[str] | None = None,
) -> GeomBatch:
    """Collate a list of Data or HeteroData objects into a single Batch.
    Copied from torch_geometric.loader.dataloader.Collater.__call__.
    """
    return GeomBatch.from_data_list(
        batch,
        follow_batch=follow_batch,
        exclude_keys=exclude_keys,
    )


def update_collate_fn_map():
    """Add collate function for tensordict to the default collate function."""
    from torch.utils.data._utils.collate import default_collate_fn_map

    default_collate_fn_map.update(
        {
            TensorDict: collate_tensor_dict,
            # since the type must match exactly, we need to add BaseData and its
            # two subclasses separately
            GeomData: collate_torch_geom,
            GeomHeteroData: collate_torch_geom,
            GeomBaseData: collate_torch_geom,
        }
    )


update_collate_fn_map()
