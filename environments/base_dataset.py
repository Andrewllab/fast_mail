from __future__ import annotations

import itertools
import logging
import math
import os
import pickle
import re
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import MutableMapping, Sequence, TypeVar, cast

import h5py
import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf
from tensordict import TensorDict
from torch.utils.data import Dataset

from environments.specs import (
    CameraSpec,
    DataSpecs,
    DepthStream,
    PointCloudSpec,
    PointMapStream,
)
from transforms.base_transform import Compose, TransformPartialsDict, init_transforms
from utils.paths import iglob_follow_symlinks, resolve_path
from utils.pyg import index_reduced_batch, reduce_batch, unreduce_batch
from utils.trees import tree_call_method, tree_get_item, tree_map

log = logging.getLogger(__name__)


class TrajectoryDataset(Dataset, ABC):

    slices: TrajectorySlices

    TRANSFORMS_PKL_FILE = "transforms.pkl"
    TRANSFORMS_CFG_FILE = "transforms_cfg.yaml"
    SPECS_PKL_FILE = "specs.pkl"

    @property
    @abstractmethod
    def root_dir(self) -> Path:
        pass

    @property
    @abstractmethod
    def specs(self) -> DataSpecs:
        pass

    @property
    @abstractmethod
    def item_transforms(self) -> Compose:
        pass

    @property
    @abstractmethod
    def preprocess_transforms(self) -> Compose:
        pass

    @property
    @abstractmethod
    def preprocess_transforms_config(self) -> ListConfig:
        pass

    @classmethod
    def save_metadata(
        cls,
        folder: Path,
        transforms: Compose,
        transforms_cfg: ListConfig,
    ) -> None:
        assert isinstance(transforms, Compose)
        with open(folder / cls.TRANSFORMS_PKL_FILE, "wb") as f:
            pickle.dump(transforms, f)

        assert isinstance(transforms_cfg, ListConfig)
        with open(folder / cls.TRANSFORMS_CFG_FILE, "w") as f:
            OmegaConf.save(transforms_cfg, f)

    @classmethod
    def load_metadata(cls, folder: Path) -> tuple[Compose, ListConfig]:
        with open(folder / cls.TRANSFORMS_PKL_FILE, "rb") as f:
            transforms = pickle.load(f)
        if not isinstance(transforms, Compose):
            raise ValueError(f"Expected a Compose, got {type(transforms)}")

        with open(folder / cls.TRANSFORMS_CFG_FILE, "r") as f:
            transforms_cfg = OmegaConf.load(f)

        if not isinstance(transforms_cfg, ListConfig):
            raise ValueError(f"Expected a ListConfig, got {type(transforms_cfg)}")

        return transforms, transforms_cfg

    def __len__(self) -> int:
        return len(self.slices)

    @property
    def n_trajectories(self) -> int:
        return self.slices.n_trajectories

    @abstractmethod
    def __getitem__(self, idx: int) -> TensorDict:
        pass

    @abstractmethod
    def get_trajectory(self, traj_idx: int) -> TensorDict:
        pass

    @classmethod
    def save_trajectory(
        cls, traj: TensorDict, root_dir: Path, specs: DataSpecs
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def filename_keyfunc(path: Path) -> list[str | int]:
        """Key function for natural sorting of Path objects by filename. Strips
        the suffix and splits the filename into chunks of digits and non-digits.
        """
        return [
            int(chunk) if chunk.isdigit() else chunk.lower()
            for part in path.with_suffix("").parts
            for chunk in re.split(r"(\d+)", part)
        ]

    @staticmethod
    def compress_rgb_images(obs: MutableMapping, specs: DataSpecs) -> MutableMapping:
        """Convert any RGB images in the obs from float32 to uint8 to reduce memory
        footprint.
        """
        for key, spec in specs.obs.items():
            if not isinstance(spec, CameraSpec):
                continue

            for name, stream in spec.streams.items():
                if not isinstance(stream, (DepthStream, PointMapStream)):
                    # TODO: change this to isinstance(stream, (RGBStream, IntensityStream))
                    # once intensity streams are used everywhere

                    image = obs[key, name]

                    # if the stream is RGB or intensity, we need to convert it to uint8
                    if image.dtype != torch.uint8:
                        image = image.mul(255).clamp(0, 255).to(torch.uint8)
                        obs[key, name] = image

        return obs

    @staticmethod
    def reduce_pyg_data(obs: MutableMapping, specs: DataSpecs) -> MutableMapping:
        pcd_keys = [
            key for key, spec in specs.obs.items() if isinstance(spec, PointCloudSpec)
        ]

        for key in pcd_keys:
            batch = obs[key]
            obs[key] = reduce_batch(batch)
        return obs

    @staticmethod
    def unreduce_pyg_data(obs: MutableMapping, specs: DataSpecs) -> MutableMapping:
        pcd_keys = [
            key for key, spec in specs.obs.items() if isinstance(spec, PointCloudSpec)
        ]

        for key in pcd_keys:
            batch = obs[key]
            assert (
                isinstance(batch, MutableMapping) and "pos" in batch and "ptr" in batch
            )
            obs[key] = unreduce_batch(batch)

        return obs


class Hdf5Dataset(TrajectoryDataset):
    def __init__(
        self,
        root_dir: os.PathLike,
        action_seq_len: int,
        obs_seq_len: int,
        item_transforms: TransformPartialsDict | None = None,
        load_subset: int | float | Sequence[int] | None = None,
    ) -> None:
        self._root_dir = resolve_path(root_dir)

        pre_transforms, pre_transforms_cfg = self.load_metadata(self._root_dir)
        self._preprocess_transforms = pre_transforms
        self._preprocess_transforms_config = pre_transforms_cfg
        self._specs = pre_transforms[-1].specs

        files = iglob_follow_symlinks(self._root_dir, "**/*.hdf5")
        files = [file.relative_to(self._root_dir) for file in files]
        files = list(sorted(files, key=self.filename_keyfunc))
        self.files = get_subset(files, load_subset)

        self.trajs = [h5py.File(str(self._root_dir / file), "r") for file in self.files]

        traj_lengths = [len(traj["action"]) for traj in self.trajs]

        self.slices = TrajectorySlices(
            traj_lengths,
            obs_seq_len=obs_seq_len,
            action_seq_len=action_seq_len,
        )

        self.pcd_keys = [
            key
            for key, spec in self._specs.obs.items()
            if isinstance(spec, PointCloudSpec)
        ]

        if item_transforms is not None:
            log.debug("Instantiating item transforms...")
        self._item_transforms, self._specs = init_transforms(
            item_transforms, self._specs
        )

        # e.g. dataset rollout requires knowing how long the trajectories are
        # in the dataset
        self._specs.traj_lengths = self.slices.traj_lengths

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    @property
    def item_transforms(self) -> Compose:
        return self._item_transforms

    @property
    def preprocess_transforms(self) -> Compose:
        return self._preprocess_transforms

    @property
    def preprocess_transforms_config(self) -> ListConfig:
        return self._preprocess_transforms_config

    def __getitem__(self, idx: int) -> TensorDict:
        traj_idx, obs_slice, action_slice = self.slices[idx]
        traj = self.trajs[traj_idx]

        obs = {k: v for k, v in traj["obs"].items() if k not in self.pcd_keys}
        obs = tree_get_item(obs, obs_slice)

        pcds = {k: v for k, v in traj["obs"].items() if k in self.pcd_keys}
        # index_reduced_batch operates on any array-like object
        # because we only index with hdf5 datasets inside this function, we only
        # create a copy of the data that we actually use
        # however, the function creates a Data object with numpy arrays as fields
        pcds = {k: index_reduced_batch(pcd, obs_slice) for k, pcd in pcds.items()}
        # convert the numpy arrays in the Data object to torch tensors
        pcds = tree_call_method(pcds, "apply", torch.from_numpy)

        obs.update(pcds)

        action = tree_get_item(traj["action"], action_slice)
        ref_action = tree_get_item(traj["ref_action"], action_slice)

        td = TensorDict(
            {
                "obs": obs,
                "action": action,
                "ref_action": ref_action,
            }
        )

        # add a batch dimension so we can index
        td["obs"].auto_batch_size_(batch_dims=1)

        if "goal" in traj.keys():
            goal = tree_map(np.asarray, traj["goal"])
            for key, value in goal.items():
                if (
                    isinstance(value, np.ndarray)
                    and value.dtype == np.object_
                    and value.ndim == 0
                ):
                    # convert numpy bytes array to python bytes object, and
                    # then to python (utf-8) string
                    goal[key] = value.item().decode("utf-8")

            td["goal"] = goal

        relative_path = self.files[traj_idx]
        td["path"] = str(relative_path)

        td = self._item_transforms(td)
        return td

    def get_trajectory(self, traj_idx: int) -> TensorDict:
        traj = self.trajs[traj_idx]

        # recursively convert all h5py datasets to numpy arrays
        obs = tree_map(np.asarray, traj["obs"])
        action = tree_map(np.asarray, traj["action"])
        ref_action = tree_map(np.asarray, traj["ref_action"])

        td = TensorDict(
            {
                "obs": obs,
                "action": action,
                "ref_action": ref_action,
            }
        )

        td["obs"] = self.unreduce_pyg_data(td["obs"], self._specs)

        # add a batch dimension so we can index
        td["obs"].auto_batch_size_(batch_dims=1)

        if "goal" in traj.keys():
            goal = tree_map(np.asarray, traj["goal"])
            for key, value in goal.items():
                if (
                    isinstance(value, np.ndarray)
                    and value.dtype == np.object_
                    and value.ndim == 0
                ):
                    # convert numpy bytes array to python bytes object, and
                    # then to python (utf-8) string
                    goal[key] = value.item().decode("utf-8")

            td["goal"] = goal

        relative_path = self.files[traj_idx]
        td["path"] = str(relative_path)

        return td

    @classmethod
    def save_trajectory(
        cls, traj: TensorDict, root_dir: Path, specs: DataSpecs
    ) -> None:
        # remove batch dimension so we can add tensors with different leading dims
        traj["obs"].auto_batch_size_(batch_dims=0)

        traj["obs"] = cls.compress_rgb_images(traj["obs"], specs)
        traj["obs"] = cls.reduce_pyg_data(traj["obs"], specs)

        relative_path = Path(traj.pop("path").data)
        if "name" in traj:
            relative_path = relative_path.with_stem(
                relative_path.stem + "_" + str(traj.pop("name").data)
            )

        filepath = root_dir / relative_path
        filepath = filepath.with_suffix(".hdf5")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        traj.to_h5(str(filepath), compression="gzip", compression_opts=9)


class CustomHdf5Dataset(TrajectoryDataset):
    _root_dir: Path
    _specs: DataSpecs
    _item_transforms: Compose
    slices: TrajectorySlices

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def specs(self) -> DataSpecs:
        # e.g. dataset rollout requires knowing how long the trajectories are
        # in the dataset
        self._specs.traj_lengths = self.slices.traj_lengths
        return self._specs

    @property
    def item_transforms(self) -> Compose:
        return self._item_transforms

    @property
    def preprocess_transforms(self) -> Compose:
        # empty Compose transform, since this is the raw data
        return Compose(specs=self._specs)

    @property
    def preprocess_transforms_config(self) -> ListConfig:
        # empty transforms config, since this is the raw data
        return OmegaConf.create([])

    def __getitem__(self, idx: int) -> TensorDict:
        raise NotImplementedError


class MemmapDataset(TrajectoryDataset):
    def __init__(
        self,
        root_dir: os.PathLike,
        action_seq_len: int,
        obs_seq_len: int,
        item_transforms: TransformPartialsDict | None = None,
        load_subset: int | float | Sequence[int] | None = None,
    ) -> None:
        self._root_dir = resolve_path(root_dir)

        pre_transforms, pre_transforms_cfg = self.load_metadata(self._root_dir)
        self._preprocess_transforms = pre_transforms
        self._preprocess_transforms_config = pre_transforms_cfg
        self._specs = pre_transforms[-1].specs

        # each directory in the root_dir corresponds to a TensorDict, where the
        # directory structure mirrors the TensorDict structure
        files = [
            path.relative_to(self._root_dir)
            for path in self._root_dir.iterdir()
            if path.is_dir()
        ]
        files = list(sorted(files, key=self.filename_keyfunc))
        self.files = get_subset(files, load_subset)

        self.trajs = [
            TensorDict.load_memmap(self._root_dir / file, non_blocking=True)
            for file in self.files
        ]

        traj_lengths = [len(traj["action"]) for traj in self.trajs]

        self.slices = TrajectorySlices(
            traj_lengths,
            obs_seq_len=obs_seq_len,
            action_seq_len=action_seq_len,
        )

        self.pcd_keys = [
            key
            for key, spec in self._specs.obs.items()
            if isinstance(spec, PointCloudSpec)
        ]

        if item_transforms is not None:
            log.debug("Instantiating item transforms...")
        self._item_transforms, self._specs = init_transforms(
            item_transforms, self._specs
        )

        # e.g. dataset rollout requires knowing how long the trajectories are
        # in the dataset
        self._specs.traj_lengths = self.slices.traj_lengths

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    @property
    def item_transforms(self) -> Compose:
        return self._item_transforms

    @property
    def preprocess_transforms(self) -> Compose:
        return self._preprocess_transforms

    @property
    def preprocess_transforms_config(self) -> ListConfig:
        return self._preprocess_transforms_config

    def __getitem__(self, idx: int) -> TensorDict:
        traj_idx, obs_slice, action_slice = self.slices[idx]
        traj = self.trajs[traj_idx]

        obs = traj["obs"].exclude(*self.pcd_keys)

        # add a batch dimension so we can index
        obs.auto_batch_size_(batch_dims=1)
        obs = obs[obs_slice]

        pcds = traj["obs"].select(*self.pcd_keys)
        pcds = {k: index_reduced_batch(pcd, obs_slice) for k, pcd in pcds.items()}

        obs.update(pcds)

        action = traj["action"][action_slice]
        ref_action = traj["ref_action"][action_slice]

        td = TensorDict(
            {
                "obs": obs,
                "action": action,
                "ref_action": ref_action,
            }
        )

        # add a batch dimension so we can index
        td["obs"].auto_batch_size_(batch_dims=1)

        if "goal" in traj.keys():
            td["goal"] = traj["goal"]

        relative_path = self.files[traj_idx]
        td["path"] = str(relative_path)

        td = self._item_transforms(td)
        return td

    def get_trajectory(self, traj_idx: int) -> TensorDict:
        traj = self.trajs[traj_idx]

        obs = traj["obs"]
        action = traj["action"]
        ref_action = traj["ref_action"]

        td = TensorDict(
            {
                "obs": obs,
                "action": action,
                "ref_action": ref_action,
            }
        )

        td["obs"] = self.unreduce_pyg_data(td["obs"], self._specs)

        # add a batch dimension so we can index
        td["obs"].auto_batch_size_(batch_dims=1)

        if "goal" in traj.keys():
            td["goal"] = traj["goal"]

        relative_path = self.files[traj_idx]
        td["path"] = str(relative_path)

        return td

    @classmethod
    def save_trajectory(
        cls, traj: TensorDict, root_dir: Path, specs: DataSpecs
    ) -> None:
        # remove batch dimension so we can add tensors with different leading dims
        traj["obs"].auto_batch_size_(batch_dims=0)

        traj["obs"] = cls.compress_rgb_images(traj["obs"], specs)
        traj["obs"] = cls.reduce_pyg_data(traj["obs"], specs)

        relative_path = Path(traj.pop("path").data)
        if "name" in traj:
            name = traj.pop("name").data
            relative_path = relative_path.with_stem(
                relative_path.stem + "_" + str(name)
            )

        # save memmaps in a flat folder hierarchy relative to root_dir,
        # otherwise we can't distinguish between memmaps and directories
        filename = "_".join(relative_path.parts)
        filepath = root_dir / filename
        filepath = filepath.with_suffix("")  # ensure no suffix

        traj.memmap(str(filepath), num_threads=8)


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

    # TODO: add compatibility for batches of indices in `__getitems__`.

    def __init__(
        self,
        traj_lengths: Sequence[int],
        obs_seq_len: int,
        action_seq_len: int,
    ) -> None:
        self._traj_lengths = np.array(traj_lengths)
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # window size is the number of time steps in a sample from the beginning
        # of the observation to the action of the actions
        self.window_size = obs_seq_len + action_seq_len - 1

        # ignore any trajectories that are shorter than obs_seq_len + action_seq_len - 1
        self._samples_per_traj = np.maximum(
            self._traj_lengths - self.window_size + 1, 0
        )

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

    @property
    def traj_lengths(self) -> list[int]:
        return self._traj_lengths[self._samples_per_traj > 0].tolist()

    @property
    def n_trajectories(self) -> int:
        return (self._samples_per_traj > 0).sum().item()

    def to_offset(
        self, idx: int | list[int] | np.ndarray
    ) -> tuple[int | np.ndarray, int | np.ndarray]:
        # find which trajectory the idx belongs to by sorting into upper bounds
        traj_idx = np.searchsorted(self._upper_bounds, idx, side="right")

        # find offset within the trajectory by subtracting the lower bound
        offset = idx - self._lower_bounds[traj_idx]
        return traj_idx, offset

    def to_idx(
        self,
        traj_idx: int | list[int] | np.ndarray,
        offset: int | list[int] | np.ndarray,
    ) -> int | np.ndarray:
        return self._lower_bounds[traj_idx] + offset

    def __getitem__(self, idx: int) -> tuple[int, slice, slice]:
        traj_idx, offset = self.to_offset(idx)
        obs_start = offset

        # index `obs_seq_len` many observations at the start of the window
        # we always need a slice because we always expect a time dimension
        # even if we only have one observation
        obs_idx = slice(obs_start, obs_start + self.obs_seq_len)

        # return a slice of actions start at the last observation and stop at
        # the end of the window
        action_idx = slice(
            obs_start + self.obs_seq_len - 1,
            obs_start + self.window_size,
        )

        return int(traj_idx), obs_idx, action_idx

    def __repr__(self) -> str:
        return f"TrajectorySlices(traj_lengths={self.traj_lengths}, obs_seq_len={self.obs_seq_len}, action_seq_len={self.action_seq_len})"


class TrajectorySubset(TrajectoryDataset):
    r"""
    Subset of a TrajectoryDataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        intraj_indicesdices (sequence): Indices of the trajectories selected for subset
    """

    def __init__(self, dataset: TrajectoryDataset, traj_indices: Sequence[int]) -> None:
        self.dataset = dataset
        self._traj_indices = np.array(traj_indices)

        # zero out of the lengths are trajectories that are not in this subset
        traj_lengths = np.array(self.dataset.slices.traj_lengths)
        traj_lengths[
            np.isin(np.arange(len(traj_lengths)), self._traj_indices, invert=True)
        ] = 0

        self.slices = TrajectorySlices(
            traj_lengths=traj_lengths,
            obs_seq_len=self.dataset.slices.obs_seq_len,
            action_seq_len=self.dataset.slices.action_seq_len,
        )

    @property
    def traj_indices(self) -> list[int]:
        return self._traj_indices.tolist()

    @property
    def root_dir(self) -> Path:
        return self.dataset.root_dir

    @property
    def specs(self) -> DataSpecs:
        return self.dataset.specs

    @property
    def item_transforms(self) -> Compose:
        return self.dataset.item_transforms

    @property
    def preprocess_transforms(self) -> Compose:
        return self.dataset.preprocess_transforms

    @property
    def preprocess_transforms_config(self) -> ListConfig:
        return self.dataset.preprocess_transforms_config

    def _convert_idx(self, idx: int | list[int] | np.ndarray) -> int | np.ndarray:
        traj_idx, offset = self.slices.to_offset(idx)
        # check that every computed traj_idx is part of the subset
        assert (traj_idx == self._traj_indices[:, None]).any(axis=0).all()
        idx = self.dataset.slices.to_idx(traj_idx, offset)
        return idx

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self._convert_idx(i) for i in idx]]
        return self.dataset[self._convert_idx(idx)]

    def __getitems__(self, indices: list[int]) -> list:
        # add batched sampling support when parent dataset supports it.
        # see torch.utils.data._utils.fetch._MapDatasetFetcher
        if callable(getattr(self.dataset, "__getitems__", None)):
            return self.dataset.__getitems__(self._convert_idx(indices))  # type: ignore[attr-defined]
        else:
            return [self.dataset[idx] for idx in self._convert_idx(indices)]

    def get_trajectory(self, traj_idx: int) -> TensorDict:
        return self.dataset.get_trajectory(self._traj_indices[traj_idx])


def random_traj_split(
    dataset: TrajectoryDataset,
    lengths: Sequence[int] | Sequence[float],
    generator: torch.Generator | None = torch.default_generator,
) -> list[TrajectorySubset]:
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset.n_trajectories)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    Example:
        >>> # xdoctest: +SKIP
        >>> generator1 = torch.Generator().manual_seed(42)
        >>> generator2 = torch.Generator().manual_seed(42)
        >>> random_split(range(10), [3, 7], generator=generator1)
        >>> random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: list[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(dataset.n_trajectories * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = dataset.n_trajectories - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != dataset.n_trajectories:  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )

    indices = torch.randperm(sum(lengths), generator=generator).tolist()  # type: ignore[arg-type, call-overload]
    lengths = cast(Sequence[int], lengths)
    return [
        TrajectorySubset(dataset, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]


T = TypeVar("T")


def get_subset(
    files: Sequence[T], subset: int | float | Sequence[int] | Sequence[str] | str | None
) -> Sequence[T]:
    if subset is None:
        return files

    if isinstance(subset, str) and subset.startswith("slice"):
        # parse slice notation, e.g. "slice(0, 10, 2)"
        log.info(f"Loading {subset} out of {len(files)} total trajectories found.")
        _slice = eval(subset)
        return files[_slice]

    if isinstance(subset, Sequence) and isinstance(subset[0], str):
        log.info(
            f"Loading the following trajectories out of {len(files)} total trajectories found: {subset}"
        )

        if isinstance(files[0], Path):
            return [file for file in files if file.name in subset]

        if isinstance(files[0], str):
            return [file for file in files if file in subset]

        raise NotImplementedError(
            f"Cannot select subset {subset} for items of type {type(files[0])}."
        )

    if isinstance(subset, Sequence) and isinstance(subset[0], int):
        log.info(
            f"Loading the following trajectories out of {len(files)} total trajectories found: {subset}"
        )
        return [files[i] for i in subset]

    if isinstance(subset, (int, float)):
        if isinstance(subset, float):
            # if subset is a percentage, convert it to an integer
            subset = int(len(files) * subset)

        if isinstance(subset, int):
            # do not index less than 1 file
            subset = max(1, subset)

        log.info(
            f"Loading only {subset} trajectories out of {len(files)} total trajectories found."
        )
        return files[:subset]

    raise NotImplementedError(
        f"Cannot select subset {subset} of type {type(subset)}. Please provide an integer, a float, a sequence of integers, a sequence of strings, or a slice notation string."
    )
