from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data._utils.collate import default_collate

from environments.datasets.base_dataset import TrajectoryDataset
from environments.specs import ActionSpec, CameraSpec, DataSpecs, Spec

log = logging.getLogger(__name__)


class FurnitureBenchDataset(TrajectoryDataset):
    def find_raw_files(self) -> list[Path]:
        files = list(sorted(self.root_dir.glob("*.pkl")))
        return files

    def load_from_raw_file(self, filepath: Path) -> TensorDict | list[TensorDict]:
        log.debug(f"Loading trajectory from file {filepath}")
        with open(filepath, "rb") as f:
            data: dict[str, list[np.ndarray]] = pickle.load(f)

        # remove all keys except observations and actions (furniture, rewards, skills)
        data = {key: data[key] for key in ("observations", "actions")}

        # collate lists of numpy arrays into tensors (conversion is automatic)
        data = {k: default_collate(v) for k, v in data.items()}

        # convert to TensorDict
        traj = TensorDict(data)

        # concatenate the EE position and end effector to form robot_state observation
        robot_state = torch.cat(
            (
                traj["observations", "robot_state", "ee_pos"],
                traj["observations", "robot_state", "ee_quat"],
            ),
            dim=-1,
        )

        # remap keys and convert doubles to floats
        traj = TensorDict(
            {
                "obs": {
                    "wrist_cam": traj["observations", "color_image1"],
                    "front_cam": traj["observations", "color_image2"],
                    "robot_state": robot_state.float(),
                },
                "action": traj["actions"].float(),
            },  # type: ignore
        )

        # remove observation from final time step
        # (actions have one less time step than observations)
        traj["obs"].auto_batch_size_()  # need a batch size before we can index
        traj["obs"] = traj["obs"][:-1]

        # set batch_size in TensorDict, otherwise it can't be indexed
        traj.auto_batch_size_()

        # TODO: verify shapes according to specs
        return traj

    def get_specs(self) -> DataSpecs:
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
            action=ActionSpec(shape=(self.action_seq_len, 8), type="action"),
        )
