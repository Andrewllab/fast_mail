from __future__ import annotations

import dataclasses
import logging
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Literal, Sequence, TypeVar

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import Dataset

from environments.specs import DataSpecs, load_specs, save_specs
from transforms.base_transform import (
    TransformPartialsDict,
    get_transforms_config,
    init_transforms,
    load_transforms_config,
    save_transforms_config,
)
from utils.tensordict import load_tensordict, save_tensordict

IndexType = slice | Tensor | Sequence
DeviceType = Literal["disk", "gpu"] | str | torch.device
T = TypeVar("T")

log = logging.getLogger(__name__)

SPECS_FILE = "specs.pkl"
TRANSFORMS_FILE = "transforms.yaml"


class TrajectoryDataset(Dataset, ABC):
    def __init__(
        self,
        root_dir: os.PathLike,
        action_seq_len: int,
        obs_seq_len: int,
        device: DeviceType = "disk",
        transforms: TransformPartialsDict | None = None,
        preprocess_transforms: TransformPartialsDict | None = None,
        preprocessed_dir: Path | os.PathLike | None = None,
        overwrite_preprocessed: bool = False,
        debug_preprocess: bool = False,
        load_subset: int | float | Sequence[int] | None = None,
        # filter: Callable[[Any], bool] | None = None,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len
        self.load_subset = load_subset

        if debug_preprocess and preprocess_transforms is not None:
            # TODO: implement this by running preprocessing in memory instead of saving to disk
            raise NotImplementedError

            log.debug(
                "`debug_preprocess` activated. Prepending preprocess transforms to cpu transforms."
            )
            if transforms is not None:
                transforms = {**preprocess_transforms, **transforms}
            else:
                transforms = preprocess_transforms
            preprocess_transforms = None

        if device == "gpu":
            device = "cuda"
        if device != "disk":
            device = torch.device(device)
        self.device = device

        if preprocessed_dir is not None:
            preprocessed_dir = Path(preprocessed_dir)

        if preprocess_transforms is not None:
            if preprocessed_dir is None:
                raise ValueError(
                    "preprocessed_dir must be specified if preprocess_transforms is not None"
                )

            self.handle_preprocessing(
                preprocess_transforms=preprocess_transforms,
                preprocessed_dir=preprocessed_dir,
                overwrite_preprocessed=overwrite_preprocessed,
            )

            if self.device != "disk":
                self.trajectories = [load_tensordict(f) for f in self.processed_files]

        else:
            if self.device != "disk":
                raw_files = self._find_raw_files()

                log.info(
                    f"{self.__class__.__name__}: Loading dataset from {self.root_dir} into memory ({len(raw_files)} trajectories)."
                )
                # load all trajectories into memory
                trajectories = [
                    self.load_from_raw_file(filepath) for filepath in raw_files
                ]
                if isinstance(trajectories[0], list):
                    trajectories = [
                        traj for sublist in trajectories for traj in sublist
                    ]
                self.trajectories = trajectories

                self._specs = self.get_specs()

                # collect these statistics we need for TrajectorySlices and
                # action spec
                all_actions = torch.cat(
                    [traj["action"] for traj in self.trajectories], dim=0
                )
                self.specs.action.update_stats(all_actions)

                trajectory_lengths = [
                    int(traj.batch_size[0]) for traj in self.trajectories
                ]
                self.specs.extend_lengths(trajectory_lengths)

            else:
                # cannot load all trajectories into memory, so we need to
                # convert them one by one into a file format that can be loaded
                # quickly from disk
                # we do this by "preprocessing" with a no-op transform
                if preprocessed_dir is None:
                    preprocessed_dir = self.root_dir.parent / (
                        f"{self.root_dir.name}_memmap"
                    )
                self.handle_preprocessing(
                    preprocess_transforms={},
                    preprocessed_dir=preprocessed_dir,
                    overwrite_preprocessed=True,
                )

        log.debug(
            f"Dataset trajectories have the following lengths:\n{self._specs.lengths}"
        )
        # window size is the number of time steps in a sample from the beginning
        # of the observation to the action of the actions
        window_size = action_seq_len + obs_seq_len - 1
        self.slices = TrajectorySlices(self.specs.lengths, window_size)
        log.debug(f"Dataset contains {len(self.slices)} samples in total.")

        log.debug("Instantiating cpu transforms...")
        self.transform, self._specs = init_transforms(transforms, self._specs)

        # TODO: also update obs_seq_len
        action_spec = self._specs.action
        self._specs = self._specs.replace(
            action=dataclasses.replace(action_spec, time=self.action_seq_len)
        )

        # move dataset to device if needed
        if self.device != "disk":
            self.trajectories = [traj.to(device) for traj in self.trajectories]

    @abstractmethod
    def find_raw_files(self) -> list[Path]:
        pass

    @abstractmethod
    def load_from_raw_file(self, filepath: Path) -> TensorDict | list[TensorDict]:
        """Load a trajectory from a raw data file and return it as a TensorDict.
        The TensorDict should have a single batch dimension corresponding to the
        length of the trajectory. Data without a time dimension, such as goal
        information, can be wrapped in a NonTensorData, in which case it is
        ignored by `auto_batch_size()`.
        """
        pass

    @abstractmethod
    def get_specs(self) -> DataSpecs:
        pass

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def handle_preprocessing(
        self,
        preprocess_transforms: TransformPartialsDict,
        preprocessed_dir: Path,
        overwrite_preprocessed: bool,
    ) -> None:
        """Decide if preprocessing is needed and if so, do it. After this method
        runs, either:
        - preprocessing has been completed (potentially overwriting existing)
        - an exception is raised because existing data is found but overwrite is False
        - preprocessing is not needed and we can load the existing data

        In all 3 cases, self.processed_files and self.specs have been set, but
        self.trajectories has not.
        """
        transforms_file = preprocessed_dir / TRANSFORMS_FILE
        specs_file = preprocessed_dir / SPECS_FILE

        # If these files don't exist, we assume that preprocessing has not been
        # done yet
        if not (transforms_file.exists() and specs_file.exists()):
            # delete the preprocessed directory if it exists and create a new one
            if preprocessed_dir.exists():
                if not overwrite_preprocessed:
                    raise ValueError(
                        f"Preprocessed directory {preprocessed_dir} is not empty (perhaps an incomplete earlier preprocessing run). Set overwrite_preprocessed=True to overwrite."
                    )
                else:
                    log.warning(
                        f"Preprocessed directory {preprocessed_dir} already exists. Deleting it."
                    )
                    shutil.rmtree(preprocessed_dir)

            log.info(
                f"Preprocessing dataset from {self.root_dir} and saving to {preprocessed_dir}"
            )
            self.preprocess(preprocess_transforms, preprocessed_dir)
            return

        # Some preprocessed data found, so we need to check if it matches the current
        # preprocess_transforms
        old_transforms_cfg = load_transforms_config(transforms_file)
        transforms_cfg = get_transforms_config(preprocess_transforms)
        if old_transforms_cfg != transforms_cfg:
            if not overwrite_preprocessed:
                raise ValueError(
                    f"Preprocess transforms do not match saved version at {preprocessed_dir}. Set overwrite_preprocessed=True to overwrite."
                )

            log.warning(
                f"Preprocess transforms do not match existing version in {preprocessed_dir}. Overwriting preprocessed data."
            )
            log.debug(
                f"Old config:\n{old_transforms_cfg}\n\nNew config:\n{transforms_cfg}"
            )
            shutil.rmtree(preprocessed_dir)
            self.preprocess(preprocess_transforms, preprocessed_dir)
            return

        # preprocessed data matches, so we can load it
        self._specs = load_specs(specs_file)
        self.processed_files = self._find_processed_files(preprocessed_dir)
        log.info(
            f"Loading preprocessed data from {preprocessed_dir} ({len(self.processed_files)} trajectories)."
        )

    def preprocess(
        self, preprocess_transforms: TransformPartialsDict, preprocessed_dir: Path
    ) -> None:
        """Runs the preprocessing transforms on the raw data and saves the
        resulting TensorDicts to disk. This method also sets self.specs and
        self.processed_files.
        """
        assert not preprocessed_dir.exists()
        preprocessed_dir.mkdir(parents=True)

        raw_files = self._find_raw_files()

        specs = self.get_specs()
        transforms, specs = init_transforms(preprocess_transforms, specs, wrap=False)
        assert isinstance(transforms, list)

        processed_files = []
        for raw_file in raw_files:
            trajs = self.load_from_raw_file(raw_file)
            log.debug(f"Preprocessing trajectories from file {raw_file}")
            if isinstance(trajs, list):
                filenames = [
                    preprocessed_dir / f"{raw_file.stem}_{i:03d}"
                    for i in range(len(trajs))
                ]
            else:
                trajs = [trajs]
                filenames = [preprocessed_dir / f"{raw_file.stem}"]

            for traj, filename in zip(trajs, filenames):
                # TODO: handle the case where multiple trajectories are created
                # TODO: implement multiprocessing and optionally running on GPU
                for transform in transforms:
                    traj = transform.call_trajectory(traj)

                specs.action.update_stats(traj["action"])
                specs.append_length(int(traj.batch_size[0]))

                save_tensordict(traj, filename, specs)
                processed_files.append(filename)

        # save the specs and transforms to the preprocessed directory to
        # to indicate that preprocessing completed successfully
        transforms_file = preprocessed_dir / TRANSFORMS_FILE
        specs_file = preprocessed_dir / SPECS_FILE
        save_transforms_config(preprocess_transforms, transforms_file)
        save_specs(specs, specs_file)

        self._specs = specs
        self.processed_files = processed_files

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> TensorDict:
        traj_idx, start, end = self.slices(idx)

        if self.device != "disk":
            # if dataset fits into memory, we can just index a list
            trajectory = self.trajectories[traj_idx]

        else:
            # if dataset does not fit into memory, then we have a preprocessed
            # version that we can load from disk
            trajectory = load_tensordict(self.processed_files[traj_idx], start, end)
            # since we have already sliced the tensordict, adjust start and end
            start, end = 0, None

        # create new TensorDict because we cannot inherit batch_size from
        # trajectory, as obs and action have different leading dims
        data = TensorDict(
            {
                # index `obs_seq_len` many observations at the start of the
                # window
                "obs": trajectory["obs"][start : start + self.obs_seq_len],
                # actions start at the last observation and stop at the end
                # of the window
                "action": trajectory["action"][start + self.obs_seq_len - 1 : end],
            }
        )
        if "goal" in trajectory:
            # this implicitly unwraps any NonTensorData used for storing goal
            data["goal"] = trajectory["goal"]

        # apply any transforms
        data = self.transform(data)
        return data

    def _find_raw_files(self) -> list[Path]:
        files = self.find_raw_files()

        return get_subset(files, self.load_subset)

    def _find_processed_files(self, preprocessed_dir: Path) -> list[Path]:
        """Find all processed files in the preprocessed directory."""
        # this handles the case where the processed files are actually
        # directories because memory-mapped tensordicts are saved as directories
        files = [
            path
            for path in sorted(preprocessed_dir.iterdir())
            if path.name not in (SPECS_FILE, TRANSFORMS_FILE)
        ]

        return files

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


def get_subset(
    files: Sequence[T], subset: int | float | Sequence[int] | None
) -> Sequence[T]:
    """Get a subset of files based on the specified subset percentage."""
    if subset is None:
        return files

    if isinstance(subset, (int, float)):

        if isinstance(subset, float):
            # if subset is a percentage, convert it to an integer
            subset = int(len(files) * subset)

        if isinstance(subset, int):

            # do not index less than 1 file
            subset = max(1, subset)

        log.debug(f"Loading only {subset} files out of {len(files)} total files found.")
        return files[:subset]

    log.debug(
        f"Loading the following files out of {len(files)} total files found: {subset}"
    )
    return [files[i] for i in subset]
