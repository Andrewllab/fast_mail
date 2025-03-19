import os
from typing import TYPE_CHECKING, Sequence

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from torch import Tensor


class TrajectoryDataset(Dataset):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: 0: invalid; 1: valid
    """

    def __init__(
        self,
        root_dir: os.PathLike,
        camera_names: Sequence[str],
        device,
        state_dim: int,
        obs_dim: int,
        action_dim: int,
        max_len_data: int = 256,
        window_size: int = 1,
    ):

        self.root_dir = root_dir
        self._camera_names = list(camera_names)
        self.device = device

        self._state_dim = state_dim
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self.max_len_data = max_len_data
        self.window_size = window_size

    @property
    def camera_names(self) -> list[str]:
        return self._camera_names

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def all_actions(self) -> Tensor:
        raise NotImplementedError

    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

    def get_all_actions(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        raise NotImplementedError

    def get_all_observations(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        raise NotImplementedError
