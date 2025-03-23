from __future__ import annotations

from typing import TYPE_CHECKING

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

    @property
    def obs_space(self) -> dict:
        raise NotImplementedError

    @property
    def action_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def goal_embed_seq_len(self) -> int:
        try:
            goal_embed_space = self.obs_space["goal_embed"]
        except KeyError:
            return 0

        return goal_embed_space["shape"][0]

    @property
    def goal_embed_dim(self) -> int:
        try:
            goal_embed_space = self.obs_space["goal_embed"]
        except KeyError:
            return 0

        return goal_embed_space["shape"][-1]

    @property
    def state_dim(self) -> int:
        try:
            robot_state_space = self.obs_space["robot_state"]
        except KeyError:
            return 0

        return robot_state_space["shape"][-1]

    @property
    def action_seq_len(self) -> int:
        assert len(self.action_shape) == 2
        return self.action_shape[0]

    @property
    def action_dim(self) -> int:
        assert len(self.action_shape) == 2
        return self.action_shape[1]

    @property
    def all_actions(self) -> Tensor:
        raise NotImplementedError

    def get_all_observations(self) -> Tensor:
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        raise NotImplementedError

    def get_all_actions(self) -> Tensor:
        raise NotImplementedError
