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
from environments.dataset.base_dataset import TrajectoryDataset
from agents.utils.sim_path import sim_framework_path

log = logging.getLogger(__name__)


class PushTDataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        device="cpu",
        obs_dim: int = 32,
        action_dim: int = 2,
        state_dim: int = 5,
        max_len_data: int = 136,
        window_size: int = 1,
        relative: bool = False,
        start_idx: int = 0,
        traj_per_task: int = 1,
        # one_slice_per_traj: bool = False,
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
        self.states = torch.load(os.path.join(self.data_directory, "states.pth"))
        if relative:
            self.actions = torch.load(os.path.join(self.data_directory, "rel_actions.pth"))
        else:
            self.actions = torch.load(os.path.join(self.data_directory, "abs_actions.pth"))

        # Length of each trajectory - total 206 trajectories
        with open(os.path.join(self.data_directory, "seq_lengths.pkl"), "rb") as f:
            self.seq_lengths = pickle.load(f)

        # TODO Cut data from start_idx to start_idx + traj_per_task
        self.seq_lengths = self.seq_lengths[start_idx : start_idx + traj_per_task]
        self.states = self.states[start_idx : start_idx + traj_per_task]
        self.actions = self.actions[start_idx : start_idx + traj_per_task]
        self.agentview_rgbs = self.get_rbg_collection(start_idx, start_idx + traj_per_task)

        self.states = self.states.to(device).float()    # (N, T, state_dim) - State of T-shape
        self.actions = self.actions.to(device).float()  # (N, T, action_dim)
        self.masks = torch.ones_like(self.actions)
        n = len(self.states)

        # Align with libero dataset
        for i in range(n):
            T = self.seq_lengths[i]
            self.actions[i, T:] = 0  # redo zero padding
            self.masks[i, T:] = 0

        # Turn full trajectories into slices
        # self.one_slice_per_traj = one_slice_per_traj
        self.slices = self.get_slices()

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_rbg_collection(self, start_index, end_index):
        rgb_collection = [self.get_rgb(i, range(self.get_seq_length(i))) for i in range(start_index, end_index)]

        # Combine all the RGBs with new dimension
        return rgb_collection

    def get_rgb(self, idx, frames):
        file_path = os.path.join(self.data_directory, "obses", f"episode_{idx:03d}.pth")
        obs = torch.load(file_path)
        obs = obs[frames]  # THWC
        obs = einops.rearrange(obs, "T H W C -> T C H W") / 255.0
        return obs

    # def get_frames(self, idx, frames):
    #     obs = self.agentview_rgbs[idx][[frames][0:1]]  # Only get the image current frame
    #     act = self.actions[idx, frames]              # Get the action of the current frame and the future
    #     mask = torch.ones(len(act)).bool()
    #
    #     # Wrap in a dictionary
    #     obs_dict = {"agentview_rgb": obs, "lang_emb": False}
    #     return obs_dict, act, mask

    def get_slices(self):  #Extract sample slices that meet certain conditions
        slices = []

        min_seq_length = np.inf
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            # min_seq_length = min(T, min_seq_length) if not self.one_slice_per_traj else self.window_size

            if T - self.window_size < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={self.window_size}")
            else:
                slices += [
                    (i, start, start + self.window_size) for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    # TODO: CHECK THIS
    def get_all_observations(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid observations
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.states[i, :T, :])
        return torch.cat(result, dim=0)

    def __getitem__(self, idx):
        """
        The idx is the index of the slice, not the trajectory.
        """
        i, start, end = self.slices[idx]

        obs = self.agentview_rgbs[i][start:start + 1]  # Only get the image current frame
        act = self.actions[i, start:end]  # Get the action of the current frame and the future
        mask = self.masks[i, start:end]
        state = self.states[i, start:end]

        # Wrap in a dictionary
        obs_dict = {"agentview_rgb_image": obs, "lang_emb": False, "state": state}
        return obs_dict, act, mask

    def __len__(self):
        return len(self.slices)