from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data._utils.collate import default_collate

from environments.datasets.base_dataset import TrajectoryDataset
from environments.specs import ActionSpec, CameraSpec, DataSpecs, Spec

if TYPE_CHECKING:
    from torch import Tensor


log = logging.getLogger(__name__)


class FurnitureBenchDataset(TrajectoryDataset):
    def find_filepaths(self) -> list[Path]:
        files = list(sorted(self.root_dir.glob("*.pkl")))
        return files

    def load_trajectory_from_file(self, filepath: Path) -> TensorDict:
        log.debug(f"Loading trajectory from file {filepath}")
        with open(filepath, "rb") as f:
            data: dict[str, list[np.ndarray]] = pickle.load(f)
        return prepare_trajectory(data)

    def get_specs(self, all_actions: Tensor | None = None) -> DataSpecs:
        return DataSpecs(
            obs={
                "wrist_cam": CameraSpec(
                    shape=(self.obs_seq_len, 224, 224, 3), type="rgb"
                ),
                "front_cam": CameraSpec(
                    shape=(self.obs_seq_len, 224, 224, 3), type="rgb"
                ),
                "robot_state": Spec(shape=(self.obs_seq_len, 7), type="state"),
            },
            action=ActionSpec(
                shape=(self.action_seq_len, 8), type="action", all_actions=all_actions
            ),
        )


def prepare_trajectory(data: dict[str, list[np.ndarray]]) -> TensorDict:
    # TODO: verify shapes according to specs
    # TODO: simplify this function

    # remove all keys except observations and actions (furniture, rewards, skills)
    data = {key: data[key] for key in ("observations", "actions")}

    # collate lists of numpy arrays into tensors (conversion is automatic)
    data = {k: default_collate(v) for k, v in data.items()}

    # convert to TensorDict
    td_data = TensorDict(data)

    # concatenate the EE position and end effector to form robot_state observation
    td_data["observations", "robot_state"] = (
        td_data["observations", "robot_state"]
        .select("ee_pos", "ee_quat", inplace=True)
        .cat_from_tensordict(dim=-1, sorted=("ee_pos", "ee_quat"))
    )

    # remove observation from final time step
    # (actions have one less time step than observations)
    for key, value in td_data["observations"].items():
        td_data["observations", key] = value[:-1]

    # remap keys
    td_data["obs"] = td_data.pop("observations")
    td_data["obs", "wrist_cam"] = td_data.pop(("obs", "color_image1"))
    td_data["obs", "front_cam"] = td_data.pop(("obs", "color_image2"))
    td_data["action"] = td_data.pop("actions")

    # convert doubles to floats
    td_data["obs", "robot_state"] = td_data["obs", "robot_state"].float()
    td_data["action"] = td_data["action"].float()

    # set batch_size in TensorDict, otherwise it can't be indexed
    td_data.auto_batch_size_()

    return td_data
