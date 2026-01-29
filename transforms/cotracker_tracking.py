from __future__ import annotations

from typing import List, Sequence

import torch
import tqdm
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
        camera_keys: str | List[str] = "left_cam",
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
        self.track_len = specs.action_seq_len + 1
        self.camera_keys = camera_keys
        if isinstance(self.camera_keys, str):
            self.camera_keys = [self.camera_keys]
        self.track_out_key = track_out_key
        self.visibility_out_key = visibility_out_key

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        for camera_key in self.camera_keys:
            traj_len = tensordict["action"].shape[0]
            video = (
                tensordict["obs"][camera_key]["rgb"].permute(0, 3, 1, 2).float() / 255.0
            )

            if "masked" in self.track_mode:
                # boolean mask
                mask = (
                    tensordict["obs"][camera_key][self.mask_key].to(self.device).float()
                )  # (1, T, 1, H, W)

            img_height, img_width = video.shape[-2], video.shape[-1]

            if "grid" in self.track_mode:
                """
                The point indices for cotracker are in the last dimension in (frame, x, y) format.
                We create a grid of points spaced by self.grid_spacing pixels.
                """
                xs = torch.arange(
                    self.grid_spacing // 2,
                    img_width,
                    self.grid_spacing,
                    device=self.device,
                )
                ys = torch.arange(
                    self.grid_spacing // 2,
                    img_height,
                    self.grid_spacing,
                    device=self.device,
                )

                zeros_tensor = torch.zeros((1), dtype=torch.int64, device=self.device)
                grid_indices = torch.cartesian_prod(zeros_tensor, xs, ys)

            track_list = []
            visibility_list = []
            for start_frame in tqdm.tqdm(range(traj_len), desc="Cotracker tracking"):
                sub_video = video[start_frame : start_frame + self.track_len].to(
                    self.device
                )

                # Grid mode selection
                if self.track_mode == "grid":
                    local_grid_indices = grid_indices

                elif self.track_mode == "masked_grid":
                    sub_mask = mask[start_frame]  # H, W
                    grid_indices_mask_values = sub_mask[
                        grid_indices[:, 2], grid_indices[:, 1]
                    ]
                    local_grid_indices = grid_indices[grid_indices_mask_values > 0.5]

                elif self.track_mode == "points":
                    local_grid_indices = tensordict["obs"][camera_key][self.points_key][
                        start_frame
                    ].to(self.device)
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

                # local_grid_indices = local_grid_indices[..., [0, 2, 1]]

                pred_tracks, pred_visibility = self.cotracker(
                    sub_video[None],
                    grid_size=0,
                    queries=local_grid_indices[None].float(),
                )

                pred_tracks = pred_tracks[0].cpu()  # remove batch dim
                pred_visibility = pred_visibility[0].cpu()  # remove batch dim

                # Pad predictions if at end of trajectory
                if start_frame + self.track_len > traj_len:
                    # Pad to traj_len using the last valid prediction
                    pad_size = start_frame + self.track_len - traj_len
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

                pred_tracks = pred_tracks.permute(
                    1, 0, 2
                )  # (num_points, traj_len, 2) since jagged nested tensors want the first dim after batch to be jagged
                pred_visibility = pred_visibility.permute(
                    1, 0
                )  # (num_points, traj_len)

                track_list.append(pred_tracks)
                visibility_list.append(pred_visibility)

            # fmt: off
            tensordict["obs"][camera_key][self.track_out_key] = torch.nested.as_nested_tensor(track_list, layout=torch.jagged)
            tensordict["obs"][camera_key][self.visibility_out_key] = torch.nested.as_nested_tensor(visibility_list, layout=torch.jagged)
            # fmt: on

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Do nothing if not called during preprocessing
        return tensordict
