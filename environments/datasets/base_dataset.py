from __future__ import annotations

import logging
import os
import os.path as osp
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Sequence

import numpy as np
from tensordict import TensorDict
from torch.utils.data import Dataset, Subset
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
    def __init__(
        self,
        root_dir: Path | os.PathLike,
        # device: Literal["disk", "cpu", "gpu"] = "cpu",
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

        self.transform, self._specs = init_transforms(transforms, self._specs)

        if self.has_download:
            self._download()

        if self.has_process:
            self._process()

    @property
    @abstractmethod
    def specs(self) -> DataSpecs:
        pass

    @property
    def raw_file_names(self) -> Union[str, list[str], tuple[str, ...]]:
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading.
        """
        raise NotImplementedError

    @property
    def processed_file_names(self) -> Union[str, list[str], tuple[str, ...]]:
        r"""The name of the files in the :obj:`self.processed_dir` folder that
        must be present in order to skip processing.
        """
        raise NotImplementedError

    def download(self) -> None:
        r"""Downloads the dataset to the :obj:`self.raw_dir` folder."""
        raise NotImplementedError

    def process(self) -> None:
        r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
        raise NotImplementedError

    def get(self, idx: int) -> TensorDict:
        r"""Gets the data object at index :obj:`idx`."""
        raise NotImplementedError

    # @property
    # def raw_dir(self) -> str:
    #     return osp.join(self.root, "raw")

    # @property
    # def processed_dir(self) -> str:
    #     return osp.join(self.root, "processed")

    @property
    def raw_paths(self) -> list[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading.
        """
        files = self.raw_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.raw_dir, f) for f in to_list(files)]

    @property
    def processed_paths(self) -> list[str]:
        r"""The absolute filepaths that must be present in order to skip
        processing.
        """
        files = self.processed_file_names
        # Prevent a common source of error in which `file_names` are not
        # defined as a property.
        if isinstance(files, Callable):
            files = files()
        return [osp.join(self.processed_dir, f) for f in to_list(files)]

    @property
    def has_download(self) -> bool:
        r"""Checks whether the dataset defines a :meth:`download` method."""
        return overrides_method(self.__class__, "download")

    def _download(self):
        if files_exist(self.raw_paths):  # pragma: no cover
            return

        fs.makedirs(self.raw_dir, exist_ok=True)
        self.download()

    @property
    def has_process(self) -> bool:
        r"""Checks whether the dataset defines a :meth:`process` method."""
        return overrides_method(self.__class__, "process")

    def _process(self):
        f = osp.join(self.processed_dir, "pre_transform.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_transform
        ):
            warnings.warn(
                "The `pre_transform` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-processing technique, pass "
                "`force_reload=True` explicitly to reload the dataset."
            )

        f = osp.join(self.processed_dir, "pre_filter.pt")
        if osp.exists(f) and torch.load(f, weights_only=False) != _repr(
            self.pre_filter
        ):
            warnings.warn(
                "The `pre_filter` argument differs from the one used in "
                "the pre-processed version of this dataset. If you want to "
                "make use of another pre-fitering technique, pass "
                "`force_reload=True` explicitly to reload the dataset."
            )

        if not self.force_reload and files_exist(self.processed_paths):
            return

        log.info("Processing...", file=sys.stderr)

        fs.makedirs(self.processed_dir, exist_ok=True)
        self.process()

        path = osp.join(self.processed_dir, "pre_transform.pt")
        fs.torch_save(_repr(self.pre_transform), path)
        path = osp.join(self.processed_dir, "pre_filter.pt")
        fs.torch_save(_repr(self.pre_filter), path)

    def __getitem__(self, idx: int | IndexType) -> Dataset | TensorDict:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices.
        """
        try:
            i = int(idx)
        except ValueError:
            return self.index_select(idx)

        data = self.get(i)
        data = data if self.transform is None else self.transform(data)
        return data

    def __iter__(self) -> Iterator[TensorDict]:
        for i in range(len(self)):
            yield self[i]

    def index_select(self, idx: IndexType) -> Subset[TrajectoryDataset]:
        r"""Creates a subset of the dataset from specified indices :obj:`idx`.
        Indices :obj:`idx` can be a slicing object, *e.g.*, :obj:`[2:5]`, a
        list, a tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type
        long or bool.
        """
        return Subset(self, idx)

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
