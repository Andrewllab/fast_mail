import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Sequence

import h5py
import numpy as np
import torch
import torch_geometric as pyg
from h5py import Group
from tensordict import TensorDict

from environments.base_dataset import (
    CustomHdf5Dataset,
    MemmapDataset,
    TrajectorySlices,
    get_subset,
    unreduce_pyg_data,
)
from environments.specs import (
    ActionSpec,
    CameraSpec,
    DataSpecs,
    DepthStream,
    ObsSpec,
    PinholeCameraIntrinsic,
    PointCloudSpec,
    RGBStream,
)
from transforms.base_transform import TransformPartialsDict, init_transforms
from utils.hdf5_utils import recursive_hdf5_to_dict
from utils.paths import iglob_follow_symlinks, resolve_path
from utils.pyg import index_reduced_batch

log = logging.getLogger(__name__)


class RoboCasaDataset(CustomHdf5Dataset):
    def __init__(
        self,
        root_dir: os.PathLike,
        action_seq_len: int,
        obs_seq_len: int,
        subfolder_name: str | None,
        item_transforms: TransformPartialsDict | None = None,
        load_subset: int | float | Sequence[int] | None = None,
    ):
        self._root_dir = resolve_path(root_dir)
        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len

        # no need to sort
        if subfolder_name is not None:
            self.files = list(
                iglob_follow_symlinks(self._root_dir, f"**/{subfolder_name}/*.hdf5")
            )
        else:
            self.files = list(iglob_follow_symlinks(self._root_dir, "**/*.hdf5"))

        if not self.files:
            raise FileNotFoundError(
                f"No raw files found in {self._root_dir}. Please check the path."
            )

        self.trajs: list[tuple[Path, str, Group]] = []
        for file in self.files:
            with h5py.File(str(file), "r") as h5_file:
                h5_dict = recursive_hdf5_to_dict(h5_file)

                # Get demo name from the foldername two levels up
                demo_name = file.parent.parent.name
                traj = self.convert_pakt_dict_to_actions_obs(h5_dict)

                if traj:
                    self.trajs.append(
                        (file.relative_to(self._root_dir), demo_name, traj)
                    )

        if not self.trajs:
            raise ValueError(f"Subset {load_subset} resulted in zero trajectories.")

        self.slices = TrajectorySlices(
            traj_lengths=[traj["actions"].shape[1] for _, _, traj in self.trajs],
            obs_seq_len=obs_seq_len,
            action_seq_len=action_seq_len,
        )

        self._load_specs()

        if item_transforms is not None:
            log.debug("Instantiating item transforms...")
        self._item_transforms, self._specs = init_transforms(
            item_transforms, self._specs
        )

    @staticmethod
    def traj_name_keyfunc(key: str) -> int:
        # each trajectory is stored under a key like "demo_1", "demo_2", etc.
        match = re.fullmatch(r"demo_(\d+)", key)
        assert match is not None
        return int(match.group(1))

    def convert_pakt_dict_to_actions_obs(
        self, pakt_dict: Dict[str, np.ndarray]
    ) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        robot_action_points = pakt_dict["action_points"][...]

        # The first timestep is the starting position
        robot_obs_points = robot_action_points[:, 0]
        # The other timesteps are the actual actions
        robot_action_points = robot_action_points[:, 1:]

        cleaned_tool_points = self.clean_tool_points(pakt_dict["tool_points"])
        if cleaned_tool_points["points"].shape[0] < 1:
            return False
        tool_action_points = cleaned_tool_points["points"]
        # The first timestep is the starting position
        tool_obs_points = tool_action_points[:, 0]
        # The other timesteps are the actual actions
        tool_action_points = tool_action_points[:, 1:]

        action = np.concatenate(
            [robot_action_points, tool_action_points], axis=0
        )  # (5+N), T, 3

        obs = {
            "tool_points": self.dict_to_pyg(
                {
                    "points": tool_obs_points,
                    "features": cleaned_tool_points["features"],
                    "colors": cleaned_tool_points["colors"],
                }
            ),
            "target_points": self.dict_to_pyg(
                {
                    "points": pakt_dict["target_points"]["points"],
                    "features": pakt_dict["target_points"]["features"],
                    "colors": pakt_dict["target_points"]["colors"],
                }
            ),
            "gripper_points": pyg.data.Batch.from_data_list(
                [pyg.data.Data(pos=torch.from_numpy(robot_obs_points).float())]
            ),
        }

        return {"actions": action, "obs": obs}

    def get_trajectory(self, traj_idx: int) -> TensorDict:
        path, name, traj = self.trajs[traj_idx]

        action = traj["actions"].astype(np.float32)

        td = TensorDict(
            {
                "obs": traj["obs"],
                "action": action,
                "ref_action": action[:5],  # only robot actions as ref_action
                "path": str(path),
                "name": name,
            },  # type: ignore
        )

        # add a batch dimension so we can index
        td["obs"].auto_batch_size_(batch_dims=1)

        return td

    def _load_specs(self) -> None:
        obs_specs = {}

        obs_specs["tool_points"] = PointCloudSpec(
            feature_dim=768, color=True, time=False
        )
        obs_specs["target_points"] = PointCloudSpec(
            feature_dim=768, color=True, time=False
        )
        obs_specs["gripper_points"] = PointCloudSpec(
            feature_dim=0, color=False, time=False
        )

        # ignore action dims related to static mobile platform
        action = ActionSpec(action_dim=3, time=self.action_seq_len)

        goal_specs = {"text": ObsSpec(elem_shape=(), time=None)}

        self._specs = DataSpecs(obs=obs_specs, action=action, goal=goal_specs)

    def clean_tool_points(
        self, tool_points: Dict[str, np.ndarray]
    ) -> Dict[str, torch.Tensor]:
        cleaned_tool_points = {}

        valid_mask = tool_points["valid_mask"][...]
        valid_mask = valid_mask.all(axis=-1)

        cleaned_tool_points["points"] = tool_points["points"][...][valid_mask]
        cleaned_tool_points["features"] = tool_points["features"][...][valid_mask]
        cleaned_tool_points["colors"] = tool_points["colors"][...][valid_mask]

        return cleaned_tool_points

    def dict_to_pyg(self, pointcloud: Dict[str, np.ndarray]) -> torch.Tensor:
        points = torch.from_numpy(pointcloud["points"]).float()
        colors = (
            torch.from_numpy(pointcloud["colors"]).float() / 255.0
        )  # normalize colors
        features = torch.from_numpy(pointcloud["features"]).float()

        data = pyg.data.Batch.from_data_list(
            [pyg.data.Data(pos=points, x=features, color=colors)]
        )

        return data

    # @property
    # def specs(self) -> DataSpecs:
    #     # e.g. dataset rollout requires knowing how long the trajectories are
    #     # in the dataset
    #     self._specs.traj_lengths = self.slices_lengths
    #     return self._specs

    # @property
    # def n_trajectories(self) -> int:
    #     return len(self.slices)


class PaktMemmapDataset(MemmapDataset):
    def __init__(
        self,
        root_dir: os.PathLike,
        action_seq_len: int,
        obs_seq_len: int,
        item_transforms: TransformPartialsDict | None = None,
        load_subset: int | float | Sequence[int] | None = None,
    ) -> None:
        super().__init__(
            root_dir=root_dir,
            action_seq_len=action_seq_len,
            obs_seq_len=obs_seq_len,
            item_transforms=item_transforms,
            load_subset=load_subset,
        )

        traj_lengths = [traj["action"].shape[1] for traj in self.trajs]

        self.slices = TrajectorySlices(
            traj_lengths,
            obs_seq_len=obs_seq_len,
            action_seq_len=action_seq_len,
        )

        # e.g. dataset rollout requires knowing how long the trajectories are
        # in the dataset
        self._specs.traj_lengths = self.slices.traj_lengths

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

        action = traj["action"][:, action_slice]
        ref_action = traj["ref_action"][:, action_slice]

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

        td = self._item_transforms(td)
        return td
