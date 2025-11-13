from __future__ import annotations

import logging
import os
import pickle
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import MutableMapping, Sequence, TypeVar

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
from tree import ArrayDict
from utils.paths import iglob_follow_symlinks, resolve_path
from utils.pyg import index_reduced_batch, reduce_batch, unreduce_batch

log = logging.getLogger(__name__)


class TrajectoryDataset(Dataset, ABC):

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

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> TensorDict:
        pass

    @property
    @abstractmethod
    def n_trajectories(self) -> int:
        pass

    @abstractmethod
    def get_trajectory(self, traj_idx: int) -> TensorDict:
        pass

    def get_chunk(self, idx: slice) -> TensorDict:
        raise NotImplementedError

    @classmethod
    def save_trajectory(
        cls, traj: TensorDict, root_dir: Path, specs: DataSpecs
    ) -> None:
        raise NotImplementedError

    @classmethod
    def save_chunk(
        cls, traj: TensorDict, file: Path, idx: slice, specs: DataSpecs
    ) -> None:
        raise NotImplementedError

    @staticmethod
    def filename_keyfunc(path: Path) -> list[str | int]:
        """Key function for natural sorting of Path objects by filename."""
        return [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.stem)
        ]


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
        files = list(sorted(files, key=self.filename_keyfunc))
        self.files = get_subset(files, load_subset)

        self.trajs = [h5py.File(str(file), "r") for file in self.files]

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
        self._item_transform, self._specs = init_transforms(
            item_transforms, self._specs
        )

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def specs(self) -> DataSpecs:
        self._specs.traj_lengths = self.slices.traj_lengths
        return self._specs

    @property
    def item_transforms(self) -> Compose:
        return self._item_transform

    @property
    def preprocess_transforms(self) -> Compose:
        return self._preprocess_transforms

    @property
    def preprocess_transforms_config(self) -> ListConfig:
        return self._preprocess_transforms_config

    def __len__(self) -> int:
        return len(self.slices)

    @property
    def n_trajectories(self) -> int:
        return len(self.trajs)

    def __getitem__(self, idx: int) -> TensorDict:
        traj_idx, obs_slice, action_slice = self.slices[idx]
        traj = self.trajs[traj_idx]

        obs = {k: v for k, v in traj["obs"].items() if k not in self.pcd_keys}
        obs = ArrayDict(obs)[obs_slice].to_dict()

        pcds = {k: v for k, v in traj["obs"].items() if k in self.pcd_keys}
        # index_reduced_batch operates on any array-like object
        # because we only index with hdf5 datasets inside this function, we only
        # create a copy of the data that we actually use
        # however, the function creates a Data object with numpy arrays as fields
        pcds = {k: index_reduced_batch(pcd, obs_slice) for k, pcd in pcds.items()}
        # recursively convert all numpy arrays to torch tensors
        # this works because the Data object also has an `apply` method, so it
        # duck-types as an ArrayDict
        pcds = ArrayDict(pcds).apply(torch.from_numpy).to_dict()

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
            goal = ArrayDict(traj["goal"])[...].to_dict()
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

        relative_path = self.files[traj_idx].relative_to(self._root_dir)
        td["path"] = str(relative_path)

        td = self._item_transform(td)
        return td

    def get_trajectory(self, traj_idx: int) -> TensorDict:
        traj = self.trajs[traj_idx]

        # recursively convert all h5py datasets to numpy arrays
        obs = ArrayDict(traj["obs"])[...].to_dict()
        action = traj["action"][...]
        ref_action = traj["ref_action"][...]

        td = TensorDict(
            {
                "obs": obs,
                "action": action,
                "ref_action": ref_action,
            }
        )

        td["obs"] = unreduce_pyg_data(td["obs"], self._specs)

        # add a batch dimension so we can index
        td["obs"].auto_batch_size_(batch_dims=1)

        if "goal" in traj.keys():
            goal = ArrayDict(traj["goal"])[...].to_dict()
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

        relative_path = self.files[traj_idx].relative_to(self._root_dir)
        td["path"] = str(relative_path)

        return td

    @classmethod
    def save_trajectory(
        cls, traj: TensorDict, root_dir: Path, specs: DataSpecs
    ) -> None:
        # remove batch dimension so we can add tensors with different leading dims
        traj["obs"].auto_batch_size_(batch_dims=0)

        traj["obs"] = compress_rgb_images(traj["obs"], specs)
        traj["obs"] = reduce_pyg_data(traj["obs"], specs)

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
    _item_transform: Compose
    slices: TrajectorySlices
    trajs: list

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def specs(self) -> DataSpecs:
        self._specs.traj_lengths = self.slices.traj_lengths
        return self._specs

    @property
    def item_transforms(self) -> Compose:
        return self._item_transform

    @property
    def preprocess_transforms(self) -> Compose:
        return Compose(specs=self._specs)

    @property
    def preprocess_transforms_config(self) -> ListConfig:
        return OmegaConf.create([])

    def __len__(self) -> int:
        return len(self.slices)

    @property
    def n_trajectories(self) -> int:
        return len(self.trajs)

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
        files = [path for path in self._root_dir.iterdir() if path.is_dir()]
        files = list(sorted(files, key=self.filename_keyfunc))
        self.files = get_subset(files, load_subset)

        self.trajs = [
            TensorDict.load_memmap(file, non_blocking=True) for file in self.files
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
        self._item_transform, self._specs = init_transforms(
            item_transforms, self._specs
        )

    @property
    def root_dir(self) -> Path:
        return self._root_dir

    @property
    def specs(self) -> DataSpecs:
        self._specs.traj_lengths = self.slices.traj_lengths
        return self._specs

    @property
    def item_transforms(self) -> Compose:
        return self._item_transform

    @property
    def preprocess_transforms(self) -> Compose:
        return self._preprocess_transforms

    @property
    def preprocess_transforms_config(self) -> ListConfig:
        return self._preprocess_transforms_config

    def __len__(self) -> int:
        return len(self.slices)

    @property
    def n_trajectories(self) -> int:
        return len(self.trajs)

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

        relative_path = self.files[traj_idx].relative_to(self._root_dir)
        td["path"] = str(relative_path)

        td = self._item_transform(td)
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

        td["obs"] = unreduce_pyg_data(td["obs"], self._specs)

        # add a batch dimension so we can index
        td["obs"].auto_batch_size_(batch_dims=1)

        if "goal" in traj.keys():
            td["goal"] = traj["goal"]

        relative_path = self.files[traj_idx].relative_to(self._root_dir)
        td["path"] = str(relative_path)

        return td

    @classmethod
    def save_trajectory(
        cls, traj: TensorDict, root_dir: Path, specs: DataSpecs
    ) -> None:
        # remove batch dimension so we can add tensors with different leading dims
        traj["obs"].auto_batch_size_(batch_dims=0)

        traj["obs"] = compress_rgb_images(traj["obs"], specs)
        traj["obs"] = reduce_pyg_data(traj["obs"], specs)

        relative_path = Path(traj.pop("path").data)
        if "name" in traj:
            relative_path = relative_path.with_stem(
                relative_path.stem + "_" + str(traj.pop("name").data)
            )

        # save memmaps in a flat folder hierarchy relative to root_dir,
        # otherwise we can't distinguish between memmaps and directories
        filepath = root_dir / relative_path.name
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
        self.traj_lengths = traj_lengths
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

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

    def __getitem__(self, idx: int) -> tuple[int, slice, slice]:
        # find which trajectory the idx belongs to by sorting into upper bounds
        traj_idx = np.searchsorted(self._upper_bounds, idx, side="right")

        # find offset within the trajectory by subtracting the lower bound
        obs_start = idx - self._lower_bounds[traj_idx]

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


T = TypeVar("T")


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

        log.debug(
            f"Loading only {subset} trajectories out of {len(files)} total trajectories found."
        )
        return files[:subset]

    log.debug(
        f"Loading the following trajectories out of {len(files)} total trajectories found: {subset}"
    )
    return [files[i] for i in subset]


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


def reduce_pyg_data(obs: MutableMapping, specs: DataSpecs) -> MutableMapping:
    pcd_keys = [
        key for key, spec in specs.obs.items() if isinstance(spec, PointCloudSpec)
    ]

    for key in pcd_keys:
        batch = obs[key]
        obs[key] = reduce_batch(batch)
    return obs


def unreduce_pyg_data(obs: MutableMapping, specs: DataSpecs) -> MutableMapping:
    pcd_keys = [
        key for key, spec in specs.obs.items() if isinstance(spec, PointCloudSpec)
    ]

    for key in pcd_keys:
        batch = obs[key]
        assert isinstance(batch, MutableMapping) and "pos" in batch and "ptr" in batch
        obs[key] = unreduce_batch(batch)

    return obs
