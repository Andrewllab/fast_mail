# obs_dict, action, mask = data
#
# # put data on cuda
# for camera in obs_dict.keys():
#     obs_dict[camera] = obs_dict[camera].to(self.device)

import logging
import random
import pickle

import cv2
import h5py
import os
import torch
import numpy as np
import einops
from fast_mail.environments.dataset.base_dataset import TrajectoryDataset
from fast_mail.agents.utils.sim_path import sim_framework_path

log = logging.getLogger(__name__)


class PushTDataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        device="cpu",
        obs_dim: int = 32,
        action_dim: int = 7,
        state_dim: int = 45,
        max_len_data: int = 136,
        window_size: int = 1,
        relative: bool = False,
        **kwargs,
    ):
        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size,
        )
        self.data_dir = sim_framework_path(self.data_directory)
        self.obs_dim = obs_dim
        self.state_dim = state_dim

        self.relative = relative
        self.states = torch.load(self.data_directory / "states.pth")
        if relative:
            self.actions = torch.load(self.data_directory / "rel_actions.pth")
        else:
            self.actions = torch.load(self.data_directory / "abs_actions.pth")
        with open(self.data_directory / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        # TODO Cut data from start_idx to start_idx + traj_per_task
        self.states = self.states.to(device).float()
        self.actions = self.actions.to(device).float()
        self.masks = torch.ones_like(self.actions)
        n = len(self.states)

        # Align with libero dataset
        for i in range(n):
            T = self.seq_lengths[i]
            self.actions[i, T:] = 0  # redo zero padding
            self.masks[i, T:] = 0

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        vid_dir = self.data_directory / "obses"
        obs = torch.load(str(vid_dir / f"episode_{idx:03d}.pth"))
        obs = obs[frames]  # THWC
        obs = einops.rearrange(obs, "T H W C -> T 1 C H W") / 255.0  # T V C H W, 1 view
        act = self.actions[idx, frames]
        mask = self.masks[idx, frames]

        # Wrap in a dictionary
        obs_dict = {}
        obs_dict["agentview_rgb"] = obs
        obs_dict["lang_emb"] = False
        return obs, act, mask

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)