import logging
import os
from typing import Sequence

import torch

from environments.dataset.base_dataset import TrajectoryDataset

log = logging.getLogger(__name__)


class FurnitureBenchDataset(TrajectoryDataset):
    def __init__(
        self,
        root_dir: os.PathLike,
        camera_names: Sequence[str],
        device="cpu",
        obs_dim: int = 32,
        action_dim: int = 7,
        state_dim: int = 45,
        max_len_data: int = 136,
        window_size: int = 1,
        start_idx: int = 0,
        traj_per_task: int = 1,
    ):
        super().__init__(
            root_dir=root_dir,
            camera_names=camera_names,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size,
        )

        log.info("Loading dataset from {}".format(self.root_dir))

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {
            "obs": torch.randn(10, 10),
            "action": torch.randn(10, 10),
        }
