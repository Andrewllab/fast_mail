from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from torch.utils.data import Dataset

from environments.data import update_collate_fn_map

if TYPE_CHECKING:
    from environments.specs import DataSpecs


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


update_collate_fn_map()
