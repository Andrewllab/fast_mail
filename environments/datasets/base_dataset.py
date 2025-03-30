from __future__ import annotations

import logging
import os
import pickle
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.utils.data import Dataset
from torch_geometric.data import Dataset as GeomDataset

from environments.data import update_collate_fn_map
from transforms.base_transform import init_transforms, to_minimal_config

if TYPE_CHECKING:
    from torch import Tensor

    from environments.specs import DataSpecs
    from transforms.base_transform import TransformPartialsDict

    IndexType = slice | Tensor | Sequence
    DeviceType = Literal["disk", "gpu"] | str | torch.device

log = logging.getLogger(__name__)


class TrajectoryDataset(Dataset, ABC):

    def __init__(
        self,
        root_dir: Path | os.PathLike,
        action_seq_len: int,
        obs_seq_len: int,
        device: DeviceType = "disk",
        transforms: TransformPartialsDict | None = None,
        preprocess_transforms: TransformPartialsDict | None = None,
        preprocessed_dir: Path | os.PathLike | None = None,
        overwrite_preprocessed: bool = False,
        # debug_preprocess: bool = False,
        load_subset: float | None = None,
        # filter: Callable[[Any], bool] | None = None,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len

        if device == "gpu":
            device = "cuda"
        if device != "disk":
            device = torch.device(device)
        self.device = device

        # window size is the number of time steps in a sample from the beginning
        # of the observation to the action of the actions
        window_size = action_seq_len + obs_seq_len - 1

        # Look for preprocessed data in `preprocessed_dir`
        if preprocess_transforms is not None:
            if preprocessed_dir is None:
                raise ValueError(
                    "preprocessed_dir must be specified if preprocess_transforms is not None"
                )

            self.preprocessed_dir = Path(preprocessed_dir)
            self.transforms_file = self.preprocessed_dir / "transforms.yaml"
            self.specs_file = self.preprocessed_dir / "specs.pkl"

            if not (self.transforms_file.exists() and self.specs_file.exists()):
                # Preprocessed data not found, so we need to preprocess the data
                log.info(
                    f"Preprocessing training data and saving to {preprocessed_dir}"
                )
                self.preprocess()

            else:
                # Some preprocessed data found, so we need to check if it matches the current
                # preprocess_transforms

                with open(self.transforms_file, "r") as f:
                    saved_preprocess_transforms = OmegaConf.load(f)

                # Compare against the current preprocess_transforms
                cfg_preprocess_transforms = to_minimal_config(preprocess_transforms)

                if cfg_preprocess_transforms != saved_preprocess_transforms:
                    if overwrite_preprocessed:
                        log.warning(
                            "Preprocess transforms do not match saved version. Overwriting preprocessed data."
                        )
                        log.debug(
                            f"Saved:\n{saved_preprocess_transforms}\n\nNew:\n{cfg_preprocess_transforms}"
                        )
                        self.preprocess()

                    else:
                        raise ValueError(
                            "Preprocess transforms do not match saved version. Set overwrite_preprocessed=True to overwrite."
                        )
                else:
                    # preprocessed data matches, so we can load it
                    log.info(f"Loading preprocessed data from {self.preprocessed_dir}")

                    with open(self.specs_file, "rb") as f:
                        self._specs = pickle.load(f)

                    # the processed files are actually directories because memory-mapped tensordicts
                    # are saved as directories
                    self.processed_files = [
                        p for p in sorted(self.preprocessed_dir.iterdir()) if p.is_dir()
                    ]

                    if self.device != "disk":
                        # load all trajectories into memory
                        self.trajectories = [
                            self.load_trajectory_from_file(filepath)
                            for filepath in self.processed_files
                        ]

                        all_actions = torch.cat(
                            [traj["action"] for traj in self.trajectories], dim=0
                        )
                        trajectory_lengths = [
                            int(traj.batch_size[0]) for traj in self.trajectories
                        ]

        else:
            # No preprocessing needed, so we can just use the raw data
            self.preprocessed_dir = None

            self.raw_files = self.find_filepaths()

            if load_subset is not None:
                # if load_subset is specified, we only load a subset of the
                # trajectories
                end = int(len(self.raw_files) * load_subset)
                log.debug(
                    f"Loading only {end} trajectories out of {len(self.raw_files)}"
                )
                self.raw_files = self.raw_files[:end]

            log.info(
                f"Loading {self.__class__.__name__} dataset with {len(self.raw_files)} trajectories from {self.root_dir}."
            )

            if self.device != "disk":
                # load all trajectories into memory
                self.trajectories = [
                    self.load_trajectory_from_file(filepath)
                    for filepath in self.raw_files
                ]

                # collect these statistics we need for TrajectorySlices and
                # action spec
                all_actions = torch.cat(
                    [traj["action"] for traj in self.trajectories], dim=0
                )
                trajectory_lengths = [
                    int(traj.batch_size[0]) for traj in self.trajectories
                ]

            else:
                # load files one by one, collecting statistics we need for
                # TrajectorySlices and action spec
                all_actions = []
                trajectory_lengths = []
                for filepath in self.raw_files:
                    traj = self.load_trajectory_from_file(filepath)
                    all_actions.append(traj["action"])
                    trajectory_lengths.append(int(traj.batch_size[0]))
                    del traj
                all_actions = torch.cat(all_actions, dim=0)

            # get specs from subclass
            self._specs = self.get_specs(all_actions)

        # If found, serialize preprocess_transforms and compare against saved version in preprocessed_dir

        # If they match, load specs from file and skip to end

        # If not, check `overwrite_preprocessed`. If False, throw exception

        # If true, load raw data (by asking subclass), then instantiate preprocess transforms, then run preprocessing, then save results back to disk
        # (if device is disk, instantiate transforms first using specs without action stats, then process trajectories sequentially)

        # end: instantiate transforms using specs

        # after init, we need:
        # - specs
        # - transforms initialized
        # - either raw_files or processed_files or trajectories

        """
        
        

        


        
        """

        log.debug(
            f"Dataset has {len(trajectory_lengths)} trajectories with the following lengths:\n{trajectory_lengths}"
        )
        self.slices = TrajectorySlices(trajectory_lengths, window_size)
        log.debug(f"Dataset contains {len(self.slices)} samples in total.")

        log.debug("Instantiating cpu transforms...")
        self.transform, self._specs = init_transforms(transforms, self._specs)

        # move dataset to device if needed
        if self.device != "disk":
            self.trajectories = [traj.to(device) for traj in self.trajectories]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def preprocess(self) -> None:
        # do the preprocessing
        assert self.preprocessed_dir is not None
        if self.preprocessed_dir.exists():
            shutil.rmtree(str(self.preprocessed_dir))
        self.preprocessed_dir.mkdir(parents=True)

        # Save the transform configuration to a YAML file
        cfg_preprocess_transforms = to_minimal_config(preprocess_transforms)
        with open(self.transforms_file, "w") as f:
            OmegaConf.save(cfg_preprocess_transforms, f)

        # Save the specs to a pickle file
        with open(self.specs_file, "wb") as f:
            pickle.dump(self.specs, f)

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int | IndexType) -> Dataset | TensorDict:
        traj_idx, start, end = self.slices(idx)

        if self.device != "disk":
            # if dataset fits into memory, we can just index a list
            trajectory = self.trajectories[traj_idx]
        elif self.preprocessed_dir is not None:
            # if dataset does not fit into main memory but we have a
            # preprocessed version, we can load it from disk
            trajectory = TensorDict.load_memmap(self.processed_files[traj_idx])
        else:
            # if dataset does not fit into main memory and there is no
            # preprocessed version, we have to ask the subclass to access the raw data
            raw_file = self.raw_files[traj_idx]
            trajectory = self.load_trajectory_from_file(raw_file)

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

    @abstractmethod
    def find_filepaths(self) -> list[Path]:
        pass

    @abstractmethod
    def load_trajectory_from_file(self, filepath: Path) -> TensorDict:
        """Load a trajectory from a raw data file and return it as a TensorDict.
        The TensorDict should have a single batch dimension corresponding to the
        length of the trajectory. Data without a time dimension, such as goal
        information, can be wrapped in a NonTensorData, in which case it is
        ignored by `auto_batch_size()`.
        """
        pass

    @abstractmethod
    def get_specs(self, all_actions: Tensor | None = None) -> DataSpecs:
        pass

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


update_collate_fn_map()
