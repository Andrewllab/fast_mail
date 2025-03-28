import logging
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
from tensordict import TensorDict
from torch.utils.data._utils.collate import default_collate

from environments.datasets.base_dataset import TrajectoryDataset, TrajectorySlices
from environments.specs import ActionSpec, CameraSpec, DataSpecs, Spec

if TYPE_CHECKING:
    from torch import Tensor

    from transforms.base_transform import TransformPartialsDict


log = logging.getLogger(__name__)


class FurnitureBenchDataset(TrajectoryDataset):
    def __init__(
        self,
        root_dir: os.PathLike,
        action_seq_len: int,
        obs_seq_len: int,
        transforms: TransformPartialsDict | None = None,
    ):

        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len

        # window size is the number of time steps in a sample from the beginning
        # of the observation to the action of the actions
        self.window_size = action_seq_len + obs_seq_len - 1

        root_dir = Path(root_dir)
        log.info("Loading FurnitureBench dataset from {}".format(root_dir))

        demo_files = list(sorted(root_dir.glob("*.pkl")))

        self.trajectories: list[TensorDict] = []
        for demo_file in demo_files:
            with open(demo_file, "rb") as f:
                data: dict[str, list[np.ndarray]] = pickle.load(f)
            traj = prepare_trajectory(data)
            self.trajectories.append(traj)

        trajectory_lengths = [int(traj.batch_size) for traj in self.trajectories]
        self.slices = TrajectorySlices(trajectory_lengths, self.window_size)

        # concatenate all actions together for collecting statistics
        all_actions = torch.cat([data["action"] for data in self.trajectories], dim=0)

        self._base_specs = self._specs = DataSpecs(
            obs={
                "wrist_cam": CameraSpec(shape=(obs_seq_len, 224, 224, 3), type="rgb"),
                "front_cam": CameraSpec(shape=(obs_seq_len, 224, 224, 3), type="rgb"),
                "robot_state": Spec(shape=(obs_seq_len, 7), type="state"),
            },
            action=ActionSpec(
                shape=(action_seq_len, 7),
                type="action",
                a_mean=all_actions.mean(0),
                a_std=all_actions.std(0),
                a_min=all_actions.min(0).values,
                a_max=all_actions.max(0).values,
            ),
        )

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __len__(self) -> int:
        return len(self.slices)

    def get(self, idx: int) -> TensorDict:
        traj_idx, start, end = self.slices(idx)
        trajectory = self.trajectories[traj_idx]

        return TensorDict(
            {
                # index `obs_seq_len` many observations at the start of the window
                "obs": trajectory["obs"][start : start + self.obs_seq_len],
                # actions start at the last observation and stop at the end of the window
                "action": trajectory["action"][start + self.obs_seq_len - 1 : end],
            }
        )


def prepare_trajectory(data: dict[str, list[np.ndarray]]) -> TensorDict:

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

    # set batch_size
    td_data.auto_batch_size_()

    # remap keys
    td_data["obs"] = td_data.pop("observations")
    td_data["obs", "wrist_cam"] = td_data.pop(("obs", "color_image1"))
    td_data["obs", "front_cam"] = td_data.pop(("obs", "color_image2"))
    td_data["action"] = td_data.pop("actions")

    return td_data
