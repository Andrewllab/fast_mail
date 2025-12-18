from __future__ import annotations

from typing import Sequence

import torch
from cotracker.utils.visualizer import Visualizer
from tensordict import TensorDict
from torch_geometric.data import Data

from environments.specs import DataSpecs
from transforms.base_transform import Transform


class CotrackerPointTrackingTransform(Transform):
    """Apply a random translation and rotation to the entire pointcloud
    trajectory during preprocessing, which is then constant throughout
    training.

    Args:
        specs (DataSpecs): The data specifications.
    """

    def __init__(
        self,
        specs: DataSpecs,
        device: str | torch.device = "cuda",
        track_mode: str = "grid",
        camera_key: str = "left_cam",
        track_out_key: str = "cotracker_tracks",
        visibility_out_key: str = "cotracker_visibility",
        # Grid tracking mode parameters
        grid_spacing: int | None = None,  # in pixels
        # masked grid tracking mode parameters
        mask_key: str | None = None,
        # Points tracking mode parameters
        points_key: str | None = None,
    ):
        super().__init__()

        self.cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_offline"
        ).to(device)

        self._specs = specs
        self.device = device

        # Validate track_mode
        if track_mode not in ["grid", "masked_grid", "points"]:
            raise ValueError(
                f"Invalid track_mode {track_mode}. Must be one of ['grid', 'masked_grid', 'points']"
            )
        self.track_mode = track_mode

        # Grid spacing settings
        if track_mode == "grid" and grid_spacing is None:
            raise ValueError("grid_spacing must be provided for grid tracking mode")
        self.grid_spacing = grid_spacing

        # Mask settings
        if track_mode == "masked_grid" and mask_key is None:
            raise ValueError("mask_key must be provided for masked_grid tracking mode")
        self.mask_key = mask_key

        # Points settings
        if track_mode == "points" and points_key is None:
            raise ValueError("points_key must be provided for points tracking mode")
        self.points_key = points_key

        # General settings
        self.action_seq_len = specs.action_seq_len
        self.camera_key = camera_key
        self.track_out_key = track_out_key
        self.visibility_out_key = visibility_out_key

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        traj_len = tensordict["action"].shape[0]
        video = (
            tensordict["obs"][self.camera_key]["rgb"]
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .to(self.device)
            .float()
        )

        if "masked" in self.track_mode:
            # boolean mask
            mask = (
                tensordict["obs"][self.camera_key][self.mask_key]
                .unsqueeze(0)
                .to(self.device)
                .float()
            )  # (1, T, 1, H, W)

        img_height, img_width = video.shape[-2], video.shape[-1]

        if "grid" in self.track_mode:
            """
            The point indices for cotracker are in the last dimension in (frame, y, x) format.
            We create a grid of points spaced by self.grid_spacing pixels.
            """
            ys = torch.arange(
                self.grid_spacing // 2,
                img_height,
                self.grid_spacing,
                device=self.device,
            )
            xs = torch.arange(
                self.grid_spacing // 2,
                img_width,
                self.grid_spacing,
                device=self.device,
            )

            zeros_tensor = torch.zeros((1), dtype=torch.int64, device=self.device)
            grid_indices = torch.cartesian_prod(zeros_tensor, ys, xs)

        track_list = []
        visibility_list = []
        for start_frame in range(traj_len):
            sub_video = video[
                :, start_frame : start_frame + self.action_seq_len, :, :, :
            ]

            # Grid mode selection
            if self.track_mode == "grid":
                local_grid_indices = grid_indices

            elif self.track_mode == "masked_grid":
                sub_mask = mask[start_frame]
                grid_indices_mask_values = sub_mask[
                    0, :, grid_indices[:, 1], grid_indices[:, 2]
                ]
                local_grid_indices = grid_indices[grid_indices_mask_values > 0.5]

            elif self.track_mode == "points":
                local_grid_indices = tensordict["obs"][self.camera_key][
                    self.points_key
                ][start_frame].to(self.device)
                if local_grid_indices.shape[-1] != 3:
                    # (N, 2) -> (N, 3)
                    zeros_tensor = torch.zeros(
                        (local_grid_indices.shape[0], 1),
                        dtype=local_grid_indices.dtype,
                        device=local_grid_indices.device,
                    )
                    local_grid_indices = torch.cat(
                        [zeros_tensor, local_grid_indices], dim=-1
                    )

            pred_tracks, pred_visibility = self.cotracker(
                sub_video, grid_size=0, queries=local_grid_indices[None]
            )

            pred_tracks = pred_tracks[0]  # remove batch dim
            pred_visibility = pred_visibility[0]  # remove batch dim

            # Pad predictions if at end of trajectory
            if start_frame + self.action_seq_len > traj_len:
                # Pad to traj_len using the last valid prediction
                pad_size = start_frame + self.action_seq_len - traj_len
                pred_tracks = torch.cat(
                    [
                        pred_tracks,
                        pred_tracks[-1].repeat(pad_size, 1, 1),
                    ],
                    dim=0,
                )
                pred_visibility = torch.cat(
                    [
                        pred_visibility,
                        pred_visibility[-1].repeat(pad_size, 1),
                    ],
                    dim=0,
                )

            track_list.append(pred_tracks)
            visibility_list.append(pred_visibility)

        # fmt: off
        tensordict["obs"][self.camera_key][self.track_out_key] = torch.stack(track_list)
        tensordict["obs"][self.camera_key][self.visibility_out_key] = torch.stack(visibility_list)
        # fmt: on

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Do nothing if not called during preprocessing
        return tensordict
