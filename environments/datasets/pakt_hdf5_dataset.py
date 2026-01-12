import os
from logging import log
from typing import Mapping

import h5py
import numpy as np
import torch
from git import Sequence
from tensordict import TensorDict

from environments.base_dataset import Hdf5Dataset, TrajectorySlices, get_subset
from environments.specs import NestedTensorSpec, PointCloudSpec
from transforms.base_transform import TransformPartialsDict, init_transforms
from tree.array_dict import ArrayDict
from utils.hdf5_utils import recursive_hdf5_to_dict
from utils.nested import unpack_nested_from_storage
from utils.paths import iglob_follow_symlinks, resolve_path
from utils.pyg import index_reduced_batch


class PaktHDF5Dataset(Hdf5Dataset):
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
        self.trajs = [recursive_hdf5_to_dict(traj) for traj in self.trajs]
        self.trajs = [TensorDict(traj) for traj in self.trajs]

        self.trajs = [unpack_nested_from_storage(traj) for traj in self.trajs]

        traj_lengths = [len(traj["action"]) for traj in self.trajs]

        self.slices = TrajectorySlices(
            traj_lengths,
            obs_seq_len=1,
            action_seq_len=1,
        )

        self.pcd_keys = [
            key
            for key, spec in self._specs.obs.items()
            if isinstance(spec, PointCloudSpec)
        ]

        self.nested_keys = [
            key
            for key, spec in self._specs.obs.items()
            if isinstance(spec, NestedTensorSpec)
        ]

        if item_transforms is not None:
            log.debug("Instantiating item transforms...")
        self._item_transforms, self._specs = init_transforms(
            item_transforms, self._specs
        )

        # e.g. dataset rollout requires knowing how long the trajectories are
        # in the dataset
        self._specs.traj_lengths = self.slices.traj_lengths

    def __getitem__(self, idx: int) -> TensorDict:
        traj_idx, obs_slice, action_slice = self.slices[idx]
        traj = self.trajs[traj_idx]

        obs = {k: v for k, v in traj["obs"].items() if k not in self.pcd_keys}
        obs = ArrayDict(obs)[obs_slice.start].to_dict()

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

        for key in self.nested_keys:
            sub_keys = obs[key].keys()
            for sub_key in sub_keys:
                obs[key][sub_key] = torch.nested.as_nested_tensor(
                    [obs[key][sub_key]], layout=torch.jagged
                )

        action = torch.nested.as_nested_tensor(
            traj["action"][action_slice.start][None], layout=torch.jagged
        )
        ref_action = torch.nested.as_nested_tensor(
            traj["ref_action"][action_slice.start][None], layout=torch.jagged
        )
        phantom_action = action.clone()

        td = TensorDict(
            {
                "obs": obs,  # add batch dimension
                "action": action,
                "ref_action": ref_action,
                "phantom_action": phantom_action,
            }
        )

        # add a batch dimension so we can index
        td["obs"].auto_batch_size_(batch_dims=1)

        if "goal" in traj.keys():
            if isinstance(traj["goal"], Mapping):
                goal = ArrayDict(traj["goal"]).to_dict()
            else:
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

        td = self._item_transforms(td)
        return td
