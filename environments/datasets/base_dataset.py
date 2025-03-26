from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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

    @property
    def collate_fn(self):
        # TensorDict can already handle batched indices, so there is no need for collation
        # https://pytorch.org/tensordict/stable/tutorials/data_fashion.html#dataloaders
        return lambda x: x

    def get_all_observations(self) -> Tensor:
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        raise NotImplementedError

    def get_all_actions(self) -> Tensor:
        raise NotImplementedError
