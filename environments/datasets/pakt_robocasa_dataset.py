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

from environments.base_dataset import CustomHdf5Dataset, TrajectorySlices, get_subset
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
from utils.paths import iglob_follow_symlinks, resolve_path

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

                # Get demo name from the foldername two levels up
                demo_name = file.parent.parent.name
                traj = {
                    "actions": h5_file["action_points"][...].transpose(1, 0, 2),
                    "obs": {
                        "target_points": self.clean_target_points(
                            h5_file["target_points"]
                        ),
                        "tool_points": self.clean_tool_points(h5_file["tool_points"]),
                    },
                }

                self.trajs.append((file.relative_to(self._root_dir), demo_name, traj))

        if not self.trajs:
            raise ValueError(f"Subset {load_subset} resulted in zero trajectories.")

        self.slices = TrajectorySlices(
            traj_lengths=[traj[2]["actions"].shape[0] for traj in self.trajs],
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

    def get_trajectory(self, traj_idx: int) -> TensorDict:
        path, name, traj = self.trajs[traj_idx]

        action = traj["actions"].astype(np.float32)

        td = TensorDict(
            {
                "obs": traj["obs"],
                "action": action,
                "ref_action": action.copy(),
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
            feature_dim=768, color=True, time=self.action_seq_len
        )
        obs_specs["target_points"] = PointCloudSpec(
            feature_dim=768, color=True, time=None
        )

        # ignore action dims related to static mobile platform
        action = ActionSpec(action_dim=5, time=self.action_seq_len)

        goal_specs = {"text": ObsSpec(elem_shape=(), time=None)}

        self._specs = DataSpecs(obs=obs_specs, action=action, goal=goal_specs)

    def clean_tool_points(self, tool_points: Group) -> None:
        cleaned_tool_points = {}

        valid_mask = tool_points["valid_mask"][...]
        valid_mask = valid_mask.all(axis=-1)

        cleaned_tool_points["points"] = tool_points["points"][...][
            valid_mask
        ].transpose(1, 0, 2)
        cleaned_tool_points["features"] = tool_points["features"][...][valid_mask]
        cleaned_tool_points["colors"] = tool_points["colors"][...][valid_mask]

        return self.pointcloud_to_pyg(cleaned_tool_points)

    def clean_target_points(self, target_points: Group) -> None:
        cleaned_target_points = {}

        cleaned_target_points["points"] = target_points["points"][...]
        cleaned_target_points["features"] = target_points["features"][...]
        cleaned_target_points["colors"] = target_points["colors"][...]

        return self.pointcloud_to_pyg(cleaned_target_points)

    def pointcloud_to_pyg(self, pointcloud: Dict[str, np.ndarray]) -> torch.Tensor:
        points = torch.from_numpy(pointcloud["points"]).float()
        colors = (
            torch.from_numpy(pointcloud["colors"]).float() / 255.0
        )  # normalize colors
        features = torch.from_numpy(pointcloud["features"]).float()

        data = pyg.data.Data(pos=points, x=features, color=colors)

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
