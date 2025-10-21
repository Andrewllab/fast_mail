from typing import Sequence

import tensordict
import torch
from tensordict import NonTensorStack, TensorDict, is_leaf_nontensor
from torch.utils.data._utils.collate import collate
from torch_geometric.data import Batch as GeomBatch
from torch_geometric.data import Data as GeomData
from torch_geometric.data import HeteroData as GeomHeteroData
from torch_geometric.data.data import BaseData as GeomBaseData


def collate_tensor_dict(batch: list[TensorDict], *, collate_fn_map) -> TensorDict:
    """Collate a list of TensorDicts into a single TensorDict."""
    stacked: TensorDict = torch.stack(batch, dim=0)  # type: ignore[assignment]

    # try to collate any NonTensorStack objects more intelligently
    for key, value in stacked.items(
        include_nested=True, leaves_only=True, is_leaf=is_leaf_nontensor
    ):
        if isinstance(value, NonTensorStack):

            collated = collate(value.tolist(), collate_fn_map=collate_fn_map)

            if isinstance(collated[0], GeomData):
                # because the obs TensorDict has a batch dimension, the
                # NonTensorData has a batch dimension too. Therefore, each
                # element in the stack is a list with one Data object, so
                # collate returns a list with one DataBatch object
                assert isinstance(collated, list)
                assert len(collated) == 1
                assert isinstance(collated[0], GeomBatch)
                stacked[key] = collated[0]

            else:
                # e.g. a list of strings or something
                with tensordict.set_list_to_stack(False):
                    # convert to numpy array and wrap in NonTensorData (no indexing along batch dim)
                    stacked[key] = collated

    # recompute batch size with only a single batch dimension, since tensordict
    # eagerly increases the number of batch dims when stacking
    stacked = stacked.auto_batch_size_(batch_dims=1)

    return stacked


def collate_torch_geom(
    batch: Sequence[GeomData] | Sequence[GeomHeteroData], *, collate_fn_map
) -> GeomBatch:
    """Collate a list of Data or HeteroData objects into a single Batch.
    Modified from torch_geometric.loader.dataloader.Collater.__call__.
    """
    return GeomBatch.from_data_list(batch)


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
