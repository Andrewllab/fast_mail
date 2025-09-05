from __future__ import annotations

import dataclasses
import logging
import os
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Literal, Sequence, TypeVar

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import Dataset

from environments.specs import DataSpecs
from transforms.base_transform import (
    Compose,
    NormalizingTransform,
    Transform,
    TransformConstraint,
    TransformPartialsDict,
    get_transforms_config,
    init_transforms,
    load_transforms,
    load_transforms_config,
    save_transforms,
    save_transforms_config,
)
from utils.conf import resolve_path
from utils.tensordict import BackendType, load_tensordict, save_tensordict

IndexType = slice | Tensor | Sequence
DeviceType = Literal["disk", "gpu"] | str | torch.device
T = TypeVar("T")

log = logging.getLogger(__name__)

TRANSFORMS_CFG_FILE = "transforms_cfg.yaml"
TRANSFORMS_PKL_FILE = "transforms.pkl"


class TrajectoryDataset(Dataset, ABC):
    preprocess_transforms: Compose
    transform: Compose

    def __init__(
        self,
        root_dir: os.PathLike,
        action_seq_len: int,
        obs_seq_len: int,
        device: DeviceType = "disk",
        backend: BackendType = "memmap",
        transforms: TransformPartialsDict | None = None,
        preprocess_transforms: TransformPartialsDict | None = None,
        preprocessed_dir: Path | os.PathLike | None = None,
        overwrite_preprocessed: bool = False,
        debug_preprocess: bool = False,
        load_subset: int | float | Sequence[int] | None = None,
        gpu_preprocess_batch_size: int | None = None,
        # filter: Callable[[Any], bool] | None = None,
    ) -> None:
        super().__init__()

        self.root_dir = resolve_path(root_dir)
        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len
        self.load_subset = load_subset
        self.gpu_preprocess_batch_size = gpu_preprocess_batch_size
        self._prepreprocessed = None

        if debug_preprocess and preprocess_transforms is not None:
            # TODO: implement this by running preprocessing in memory instead of saving to disk
            raise NotImplementedError

        if device == "gpu":
            device = "cuda"
        if device != "disk":
            device = torch.device(device)
        self.device = device

        if preprocessed_dir is not None:
            preprocessed_dir = resolve_path(preprocessed_dir)

        # strip any config nodes that aren't going to get instantiated, and
        # see if there are any transforms left
        if preprocess_transforms is not None and get_transforms_config(
            preprocess_transforms
        ):
            if preprocessed_dir is None:
                raise ValueError(
                    "preprocessed_dir must be specified if preprocess_transforms is not None"
                )

            self.handle_preprocessing(
                preprocess_partials=preprocess_transforms,
                preprocessed_dir=preprocessed_dir,
                backend=backend,
                overwrite_preprocessed=overwrite_preprocessed,
            )

            if self.device != "disk":
                log.debug(f"Moving preprocessed trajectories to {self.device=}...")
                self.trajectories = [
                    load_tensordict(f, backend=backend) for f in self.processed_files
                ]

        else:
            if self.device != "disk":
                # trajectories all fit into memory, so we can load them all at once
                # then we just need to get the specs and collect trajectory lengths
                raw_files = self._find_raw_files()

                log.info(
                    f"{self.__class__.__name__}: Loading dataset from {self.root_dir} into memory ({len(raw_files)} files)."
                )
                # load all trajectories into memory
                trajectories = [
                    traj
                    for filepath in raw_files
                    for traj in self._load_from_raw_file(filepath)
                ]
                self.trajectories = trajectories

                self._specs = self._get_specs()

                # collect trajectory lengths we need for TrajectorySlices
                trajectory_lengths = [
                    traj["obs"].batch_size[0] for traj in self.trajectories
                ]
                self.specs.extend_lengths(trajectory_lengths)
                self.preprocess_transforms = Compose()
                if self.from_prepreprocessed:
                    # if the raw data was already prepreprocessed, we need to prepend
                    # the existing prepreprocess transforms to the new ones
                    self.preprocess_transforms = Compose(
                        *(
                            list(self.prepreprocess_transforms)
                            + list(self.preprocess_transforms)
                        )
                    )

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
                    preprocess_partials={},
                    preprocessed_dir=preprocessed_dir,
                    backend=backend,
                    overwrite_preprocessed=True,
                )

        log.debug(
            f"Dataset trajectories have the following lengths:\n{self._specs.lengths}"
        )
        # TODO: add this info to the action specs
        # prechunked = self.trajectories[0]["action"].ndim == 3
        self.slices = TrajectorySlices(
            self.specs.lengths,
            obs_seq_len=obs_seq_len,
            action_seq_len=action_seq_len,
            # TODO: maybe remove this from TrajectorySlices?
            actions_prechunked=False,
        )
        log.debug(f"Dataset contains {len(self.slices)} samples in total.")

        log.debug("Instantiating cpu transforms...")
        self.transform, self._specs = init_transforms(transforms, self._specs)

        # TODO: only update if not pre-chunked, otherwise the action_seq_len is fixed
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

    @torch.no_grad()
    def _preprocess_gpu(
        self, transform: Transform, traj: TensorDict, device: torch.device
    ) -> TensorDict | Sequence[TensorDict]:
        """
        Preprocess a trajectory on the GPU, optionally in memory bounded chunks, and return the
        result back on the CPU.
        Depending on the configured GPU batch size (self.gpu_preprocess_batch_size), this method
        either:
        1. Moves the entire trajectory TensorDict to the GPU and invokes transform.call_trajectory.
        2. Splits the trajectory along the leading (time / batch) dimension into chunks of size
            self.gpu_preprocess_batch_size, applies transform to each chunk on the GPU, brings each
            transformed chunk back to CPU, and concatenates them along dim=0.
        """
        assert TransformConstraint.GPU_ONLY in transform.constraints

        # Apply a gpu_only transform to a trajectory
        if self.gpu_preprocess_batch_size is None:
            # if no batch size is given, just move the entire trajectory to the GPU
            transformed = transform.call_trajectory(traj.to(device))

            if isinstance(transformed, list):
                return [t.cpu() for t in transformed]
            else:
                return transformed.cpu()

        # Split up a trajectory into chunks that fit into GPU memory,
        # apply the transform to each chunk, and concatenate the results
        if isinstance(transform, NormalizingTransform):
            raise RuntimeError(
                f"Cannot use NormalizingTransform {transform.__class__.__name__} on gpu in chunks, "
                "as there is no way to aggregate statistics across chunks."
            )

        T = traj["obs"].shape[0]
        goal = traj.get("goal", None)

        log.info(
            f"Preprocessing trajectory of length {T} in chunks of size {self.gpu_preprocess_batch_size} on {device}."
        )
        # Process chunks one at a time to minimize memory usage
        transformed_goal = None
        for i in range(0, T, self.gpu_preprocess_batch_size):
            # Create chunk on-the-fly instead of storing all chunks
            if i + self.gpu_preprocess_batch_size < T:
                chunk = TensorDict(
                    {
                        "obs": traj["obs"][i : i + self.gpu_preprocess_batch_size],
                        "action": traj["action"][
                            i : i + self.gpu_preprocess_batch_size
                        ],
                        "ref_action": traj["ref_action"][
                            i : i + self.gpu_preprocess_batch_size
                        ],
                    }
                )
            else:
                chunk = TensorDict(
                    {
                        "obs": traj["obs"][i:],
                        "action": traj["action"][i:],
                        "ref_action": traj["ref_action"][i:],
                    }
                )

            if goal is not None:
                chunk["goal"] = goal

            # We don't call transform.call_trajectory here because we are
            # essentially operating on a batch
            new_chunk_transformed = transform(chunk.to(device)).cpu()

            # Handle goal consistency check
            if transformed_goal is None and "goal" in new_chunk_transformed:
                transformed_goal = new_chunk_transformed["goal"]
            elif "goal" in new_chunk_transformed:
                assert torch.allclose(
                    transformed_goal, new_chunk_transformed["goal"]
                ), "Goal changed between chunks, which should not happen."

            # Directly modify traj to avoid accumulating large tensors in memory
            if i + self.gpu_preprocess_batch_size < T:
                traj["obs"][i : i + self.gpu_preprocess_batch_size] = (
                    new_chunk_transformed["obs"]
                )
                traj["action"][i : i + self.gpu_preprocess_batch_size] = (
                    new_chunk_transformed["action"]
                )
                traj["ref_action"][i : i + self.gpu_preprocess_batch_size] = (
                    new_chunk_transformed["ref_action"]
                )
            else:
                traj["obs"][i:] = new_chunk_transformed["obs"]
                traj["action"][i:] = new_chunk_transformed["action"]
                traj["ref_action"][i:] = new_chunk_transformed["ref_action"]

            # Free up cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # See note in _load_from_raw_file
        if transformed_goal is not None:
            traj["goal"] = transformed_goal

        return traj

    def handle_preprocessing(
        self,
        preprocess_partials: TransformPartialsDict,
        preprocessed_dir: Path,
        backend: BackendType,
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
        transforms_cfg_file = preprocessed_dir / TRANSFORMS_CFG_FILE
        transforms_pkl_file = preprocessed_dir / TRANSFORMS_PKL_FILE

        # If these files don't exist, we assume that preprocessing has not been
        # done yet
        if not (transforms_cfg_file.exists() and transforms_pkl_file.exists()):
            # delete the preprocessed directory if it exists and create a new one
            if preprocessed_dir.is_dir() and any(preprocessed_dir.iterdir()):
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
            self.preprocess(preprocess_partials, preprocessed_dir, backend=backend)
            return

        # Some preprocessed data found, so we need to check if it matches the current
        # preprocess_partials
        # TODO: also check dataset config, e.g. action_seq_len (only if actions prechunked), and subset
        old_transforms_cfg = load_transforms_config(transforms_cfg_file)
        transforms_cfg = get_transforms_config(preprocess_partials)
        if old_transforms_cfg != transforms_cfg:
            log.debug(
                f"Old config:\n{old_transforms_cfg}\n\nNew config:\n{transforms_cfg}"
            )
            if not overwrite_preprocessed:
                raise ValueError(
                    f"Preprocess transforms do not match saved version at {preprocessed_dir}. Set overwrite_preprocessed=True to overwrite."
                )

            log.warning(
                f"Preprocess transforms do not match existing version in {preprocessed_dir}. Overwriting preprocessed data."
            )
            shutil.rmtree(preprocessed_dir)
            self.preprocess(preprocess_partials, preprocessed_dir, backend=backend)
            return

        # preprocessed data matches, so we can load it
        self.preprocess_transforms = load_transforms(transforms_pkl_file)
        assert isinstance(self.preprocess_transforms, Compose)
        self._specs = self.preprocess_transforms[-1].specs
        self.processed_files = self._find_processed_files(preprocessed_dir)
        self.backend = backend
        log.info(
            f"Loaded preprocessed data from {preprocessed_dir} ({len(self.processed_files)} trajectories)."
        )

    def preprocess(
        self,
        preprocess_partials: TransformPartialsDict,
        preprocessed_dir: Path,
        backend: BackendType,
    ) -> None:
        """Runs the preprocessing transforms on the raw data and saves the
        resulting TensorDicts to disk. This method also sets self.specs and
        self.processed_files.
        """
        preprocessed_dir.mkdir(parents=True, exist_ok=True)
        assert not any(preprocessed_dir.iterdir())

        raw_files = self._find_raw_files()

        specs = self._get_specs()
        preprocess_transforms, specs = init_transforms(preprocess_partials, specs)
        assert isinstance(preprocess_transforms, Compose)
        assert specs == preprocess_transforms[-1].specs

        if any(
            TransformConstraint.GPU_ONLY in t.constraints for t in preprocess_transforms
        ):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "At least one of the preprocessing transforms is gpu_only but no GPU is available."
                )

            # pick GPU with most free memory (fallback to cuda:0)
            if torch.cuda.device_count() == 1:
                device = torch.device("cuda:0")
            else:
                free_per_dev = []
                for d in range(torch.cuda.device_count()):
                    try:
                        free, _ = torch.cuda.mem_get_info(d)
                    except Exception:
                        free, _ = 0, 0
                    free_per_dev.append((free, d))
                _, dev_idx = max(free_per_dev)
                device = torch.device(f"cuda:{dev_idx}")

            log.info(
                f"At least one of the preprocessing transforms is gpu_only. Preprocessing will be done (partly) on {device}."
            )

        # TODO: implement multiprocessing and optionally running on GPU
        processed_files = []
        for raw_file in raw_files:
            trajs = self._load_from_raw_file(raw_file)
            log.debug(f"Preprocessing trajectories from file {raw_file}")
            if not isinstance(trajs, list):
                # if we have a single trajectory, we need to wrap it in a list
                trajs = [trajs]

            # apply the transforms in order to the trajectories, handling the
            # case where a transform returns multiple trajectories for each
            # input trajectory
            for transform in preprocess_transforms:
                next_trajs = []
                for traj in trajs:
                    if TransformConstraint.GPU_ONLY in transform.constraints:
                        transformed = self._preprocess_gpu(transform, traj, device)
                    else:
                        transformed = transform.call_trajectory(traj)

                    if isinstance(transformed, list):
                        next_trajs.extend(transformed)
                    else:
                        next_trajs.append(transformed)

                trajs = next_trajs

            if len(trajs) == 1:
                filenames = [preprocessed_dir / f"{raw_file.stem}"]
            else:
                filenames = [
                    preprocessed_dir / f"{raw_file.stem}_{i:03d}"
                    for i in range(len(trajs))
                ]

            for traj, filename in zip(trajs, filenames):
                specs.append_length(traj["obs"].batch_size[0])

                save_tensordict(traj, filename, specs, backend=backend)
                processed_files.append(filename)

        # save the specs and transforms to the preprocessed directory to
        # to indicate that preprocessing completed successfully
        transforms_cfg_file = preprocessed_dir / TRANSFORMS_CFG_FILE
        transforms_pkl_file = preprocessed_dir / TRANSFORMS_PKL_FILE
        save_transforms_config(preprocess_partials, transforms_cfg_file)

        if self.from_prepreprocessed:
            # if the raw data was already prepreprocessed, we need to prepend
            # the existing prepreprocess transforms to the new ones
            preprocess_transforms = Compose(
                *(list(self.prepreprocess_transforms) + list(preprocess_transforms))
            )

        save_transforms(preprocess_transforms, transforms_pkl_file)
        self.preprocess_transforms = preprocess_transforms
        self._specs = specs
        self.processed_files = processed_files
        self.backend = backend

    @property
    def from_prepreprocessed(self) -> bool:
        # first we check if the root directory contains a transforms.pkl file,
        # which means it is the product of prepreprocessing
        # if so, we must load the specs from there and prepend any existing
        # transforms to our own
        if self._prepreprocessed is None:
            transforms_pkl_file = self.root_dir / TRANSFORMS_PKL_FILE
            self._prepreprocessed = transforms_pkl_file.exists()

            if self._prepreprocessed:
                self.prepreprocess_transforms = load_transforms(transforms_pkl_file)
                assert isinstance(self.prepreprocess_transforms, Compose)

        return self._prepreprocessed

    def __len__(self) -> int:
        return len(self.slices)

    def __getitem__(self, idx: int) -> TensorDict:
        traj_idx, obs_idx, action_idx = self.slices(idx)

        if self.device != "disk":
            # if dataset fits into memory, we can just index a list
            trajectory = self.trajectories[traj_idx]

        else:
            # if dataset does not fit into memory, then we have a preprocessed
            # version that we can load from disk
            # TODO: optimize by loading only the relevant slice
            trajectory = load_tensordict(
                self.processed_files[traj_idx], backend=self.backend
            )
            # since we have already sliced the tensordict, adjust start and end
            obs_idx, action_idx = None, None

        # create new TensorDict because we cannot inherit batch_size from
        # trajectory, as obs and action have different leading dims
        data = TensorDict(
            {
                "obs": trajectory["obs"][obs_idx],
            }
        )

        action = trajectory["action"]
        assert isinstance(action_idx, slice)
        if action.ndim == 3:
            # prechunked
            action = action[action_idx.start]
        else:
            assert action.ndim == 2
            action = action[action_idx]
        data["action"] = action

        # ref action is never prechunked
        data["ref_action"] = trajectory["ref_action"][action_idx]

        if "goal" in trajectory:
            # this implicitly unwraps any NonTensorData used for storing goal
            data["goal"] = trajectory["goal"]

        # apply any transforms
        data = self.transform(data)
        return data

    def _find_raw_files(self) -> Sequence[Path]:
        files = self.find_raw_files()

        files = list(sorted(files, key=keyfunc))

        return get_subset(files, self.load_subset)

    def _load_from_raw_file(self, filepath: Path) -> list[TensorDict]:
        if self.from_prepreprocessed:
            trajectory = load_tensordict(filepath, backend="hdf5")
            trajectories = [trajectory.contiguous()]
        else:
            trajectories = self.load_from_raw_file(filepath)

        if not isinstance(trajectories, list):
            trajectories = [trajectories]

        for traj in trajectories:
            # obs TensorDict needs to have a batch dimension so we can index it in __getitem__
            # trajectory TensorDict should probably have no batch dimension, in case
            # we prechunk the actions and they have a different leading dimension
            traj.auto_batch_size_(batch_dims=0)
            traj["obs"].auto_batch_size_(batch_dims=1)

            # store an untransformed, unnormalized copy of the action to use as ground truth when
            # evaluating using held-out demonstration data
            traj["ref_action"] = traj["action"].clone()

        return trajectories

    def _get_specs(self) -> DataSpecs:
        # if the raw data hsa actually already been prepreprocessed, we must
        # load the specs from there, since the data may not match the original
        # format
        if self.from_prepreprocessed:
            self._specs = self.prepreprocess_transforms[-1].specs

            # the specs from the raw dataset are expected not to have lengths
            # so we need to clear them to avoid double counting
            self._specs._lengths.clear()
            return self._specs

        return self.get_specs()

    def _find_processed_files(self, preprocessed_dir: Path) -> list[Path]:
        """Find all processed files in the preprocessed directory."""
        # this handles the case where the processed files are actually
        # directories because memory-mapped tensordicts are saved as directories
        files = [path for path in preprocessed_dir.iterdir() if path.suffix == ""]

        files = sorted(files, key=keyfunc)

        return files

    def __iter__(self) -> Iterator[TensorDict]:
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        arg_repr = str(len(self)) if len(self) > 1 else ""
        return f"{self.__class__.__name__}({arg_repr})"


def keyfunc(path: Path) -> list[str | int]:
    """Key function for natural sorting of Path objects by filename."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.stem)
    ]


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

    # TODO: either restore compatibility with batches of indices or ensure
    # that only a single index is passed to __call__

    def __init__(
        self,
        traj_lengths: Sequence[int],
        obs_seq_len: int,
        action_seq_len: int,
        actions_prechunked: bool = False,
    ) -> None:
        self.traj_lengths = traj_lengths
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len
        self.actions_prechunked = actions_prechunked

        # window size is the number of time steps in a sample from the beginning
        # of the observation to the action of the actions
        self.window_size = obs_seq_len + action_seq_len - 1

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
        self._lower_bounds = np.concatenate(([0], self._upper_bounds[:-1]))

    def __len__(self) -> int:
        # with a window size of W, each trajectory gives us T-W+1 samples, as long
        # as the trajectory is not shorter than the window size
        return self._upper_bounds[-1]

    def __call__(
        self, idx: int | Sequence[int]
    ) -> tuple[np.ndarray, slice | np.ndarray, slice | np.ndarray]:
        # find which trajectory the idx belongs to by sorting into upper bounds
        traj_idx = np.searchsorted(self._upper_bounds, idx, side="right")

        # find offset within the trajectory by subtracting the lower bound
        obs_start = idx - self._lower_bounds[traj_idx]

        # index `obs_seq_len` many observations at the start of the window
        # we always need a slice because we always expect a time dimension
        # even if we only have one observation
        obs_idx = slice(obs_start, obs_start + self.obs_seq_len)

        if self.actions_prechunked:
            # if actions are prechunked, we can just return the action indices
            action_idx = obs_start + self.obs_seq_len - 1
        else:
            # if actions are not prechunked, we need to return a slice
            # actions start at the last observation and stop at the end of the window
            action_idx = slice(
                obs_start + self.obs_seq_len - 1,
                obs_start + self.window_size,
            )

        return traj_idx, obs_idx, action_idx

    def __repr__(self) -> str:
        return f"TrajectorySlices(traj_lengths={self.traj_lengths}, obs_seq_len={self.obs_seq_len}, action_seq_len={self.action_seq_len})"


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
