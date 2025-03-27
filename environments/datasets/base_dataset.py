from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict
from torch.utils.data import Dataset

from environments.specs import DataSpecs

if TYPE_CHECKING:
    from torch import Tensor


class TrajectoryDataset(Dataset, ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: 0: invalid; 1: valid
    """

    @property
    @abstractmethod
    def specs(self) -> DataSpecs:
        pass

    def get_all_observations(self) -> Tensor:
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        raise NotImplementedError

    def get_all_actions(self) -> Tensor:
        raise NotImplementedError


def collate_tensor_dict(batch: list[TensorDict], *, collate_fn_map) -> TensorDict:
    """Collate a list of TensorDicts into a single TensorDict."""
    stacked: TensorDict = torch.stack(batch, dim=0)
    # BALAZS: detect any NonTensorStack that contains a torch geometric Data
    # and turn into a Batch object
    return stacked


def update_collate_fn_map():
    """Add collate function for tensordict to the default collate function."""
    from torch.utils.data._utils.collate import default_collate_fn_map

    default_collate_fn_map.update({TensorDict: collate_tensor_dict})


update_collate_fn_map()
