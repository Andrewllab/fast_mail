from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Sequence

import numpy as np
from tensordict import TensorDict
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as GeomDataset

from environments.data import update_collate_fn_map
from transforms.base_transform import init_transforms

if TYPE_CHECKING:
    from torch import Tensor

    from environments.specs import DataSpecs
    from transforms.base_transform import TransformPartialsDict

    IndexType = slice | Tensor | Sequence

log = logging.getLogger(__name__)


class TrajectoryDataset(Dataset, ABC):

    _specs: DataSpecs
    _trajectories: list[TensorDict]

    def __init__(
        self,
        root_dir: Path | os.PathLike,
        action_seq_len: int,
        obs_seq_len: int,
        device: Literal["disk", "cpu", "cuda"] = "cpu",
        transforms: TransformPartialsDict | None = None,
        # preprocess_transform: TransformPartialsDict | None = None,
        # preprocessed_dir: Path | os.PathLike | None = None,
        # overwrite_preprocessed: bool = False,
        # debug_preprocess: bool = False,
        # load_subset: float | None = None,
        # filter: Callable[[Any], bool] | None = None,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len

        # window size is the number of time steps in a sample from the beginning
        # of the observation to the action of the actions
        window_size = action_seq_len + obs_seq_len - 1

        # we assume that each trajectory is a TensorDict with a single batch dimension
        # goal information can be wrapped in a NonTensorData to ignore batch size
        trajectory_lengths = [int(traj.batch_size[0]) for traj in self.trajectories]
        log.debug(
            f"Dataset has {len(trajectory_lengths)} trajectories with the following lengths:\n{trajectory_lengths}"
        )
        self.slices = TrajectorySlices(trajectory_lengths, window_size)
        log.debug(f"Dataset contains {len(self.slices)} samples in total.")

        self.transform, self._specs = init_transforms(transforms, self._specs)

        # move dataset to device if needed
        self._trajectories = [traj.to(device) for traj in self.trajectories]

    @property
    def trajectories(self) -> list[TensorDict]:
        return self._trajectories

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int | IndexType) -> Dataset | TensorDict:
        traj_idx, start, end = self.slices(idx)
        trajectory = self.trajectories[traj_idx]

        # create new TensorDict because we cannot inherit batch_size from trajectory
        data = TensorDict(
            {
                # index `obs_seq_len` many observations at the start of the window
                "obs": trajectory["obs"][start : start + self.obs_seq_len],
                # actions start at the last observation and stop at the end of the window
                "action": trajectory["action"][start + self.obs_seq_len - 1 : end],
            }
        )
        if "goal" in trajectory:
            # this implicitly unwraps any NonTensorData used for storing goal
            data["goal"] = trajectory["goal"]

        # apply any transforms
        data = self.transform(data)
        return data

    def __iter__(self) -> Iterator[TensorDict]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ""
        return f"{self.__class__.__name__}({arg_repr})"


class TrajectorySlices:
    """A helper class for converting indices in the Dataset's `get` method
    to a trajectory index and a start and stop index within that trajectory.
    Additionally provides the length of the dataset as the length of this
    object.

    Args:
        traj_lengths (Sequence[int]): The number of time steps in each
            trajectory in the dataset.
        window_size (int): The number of time steps that should be in each
            sample. This is affected by the action sequence length as well as
            the observation sequence length.
    """

    def __init__(self, traj_lengths: Sequence[int], window_size: int):
        self.traj_lengths = traj_lengths
        self.window_size = window_size

        self._samples_per_traj = [
            traj_length - self.window_size + 1
            for traj_length in self.traj_lengths
            if traj_length >= self.window_size
        ]

        # this array defines the upper bound of indices that correspond to each
        # trajectory
        self._upper_bounds = np.cumsum(self._samples_per_traj)
        # this array defines the lower bound of indices that correspond to each
        # trajectory
        self._lower_bounds = np.concat(([0], self._upper_bounds[:-1]))

    def __len__(self) -> int:
        # with a window size of W, each trajectory gives us T-W+1 samples, as long
        # as the trajectory is not shorter than the window size
        return self._upper_bounds[-1]

    def __call__(
        self, idx: int | Sequence[int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # find which trajectory the idx belongs to by sorting into upper bounds
        traj_idx = np.searchsorted(self._upper_bounds, idx, side="right")

        # find offset within the trajectory by subtracting the lower bound
        start = idx - self._lower_bounds[traj_idx]

        end = start + self.window_size

        return traj_idx, start, end


def _repr(obj: Any) -> str:
    if obj is None:
        return "None"
    return re.sub("(<.*?)\\s.*(>)", r"\1\2", str(obj))


update_collate_fn_map()
