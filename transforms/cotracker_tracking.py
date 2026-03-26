from __future__ import annotations

from typing import List, Sequence

import torch
import tqdm
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import Transform


class CotrackerPointTrackingTransform(Transform):
    """Run CoTracker point tracking and store results in the tensordict.

    Supports:
      - "grid": uniform grid of queries
      - "masked_grid": grid queries restricted to one-or-more masks (union for tracking, then split outputs per mask)
      - "points": explicit queries provided by `points_key`

    Notes for masked_grid with multiple masks:
      - We compute the union of all masks at each start_frame to decide which points to track.
      - We run CoTracker once on that union set of points.
      - We then split the resulting tracks/visibility back into separate outputs per mask key.
        (If a point is in multiple masks, it will appear in multiple outputs.)
    """

    def __init__(
        self,
        specs: DataSpecs,
        device: str | torch.device = "cuda",
        track_mode: str = "grid",
        camera_keys: str | List[str] = "left_cam",
        rgb_key: str = "rgb",
        track_out_keys: str | Sequence[str] = "cotracker_tracks",
        visibility_out_keys: str | Sequence[str] = "cotracker_visibility",
        # Grid tracking mode parameters
        grid_spacing: int | None = None,  # in pixels
        # masked grid tracking mode parameters
        mask_keys: str | Sequence[str] | None = None,
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
        self.mask_keys: List[str] | None
        if track_mode == "masked_grid":
            if mask_keys is None:
                raise ValueError(
                    "mask_keys must be provided for masked_grid tracking mode"
                )
            if isinstance(mask_keys, str):
                self.mask_keys = [mask_keys]
            else:
                self.mask_keys = list(mask_keys)
        else:
            self.mask_keys = None

        # Points settings
        if track_mode == "points" and points_key is None:
            raise ValueError("points_key must be provided for points tracking mode")
        self.points_key = points_key

        # General settings
        self.track_len = specs.action_seq_len + 1

        self.camera_keys = camera_keys
        if isinstance(self.camera_keys, str):
            self.camera_keys = [self.camera_keys]

        # Always store as "multiple", even if user passes a single string.
        if isinstance(track_out_keys, str):
            self.track_out_keys = [track_out_keys]
        else:
            self.track_out_keys = list(track_out_keys)

        if isinstance(visibility_out_keys, str):
            self.visibility_out_keys = [visibility_out_keys]
        else:
            self.visibility_out_keys = list(visibility_out_keys)

        # If we have multiple masks, we require multiple output keys (and vice versa).
        if self.track_mode == "masked_grid":
            assert self.mask_keys is not None
            if len(self.track_out_keys) != len(self.mask_keys):
                raise ValueError(
                    f"masked_grid requires track_out_keys to match mask_keys length. "
                    f"Got {len(self.track_out_keys)=} vs {len(self.mask_keys)=}."
                )
            if len(self.visibility_out_keys) != len(self.mask_keys):
                raise ValueError(
                    f"masked_grid requires visibility_out_keys to match mask_keys length. "
                    f"Got {len(self.visibility_out_keys)=} vs {len(self.mask_keys)=}."
                )
        else:
            # In non-masked modes we produce exactly one set of tracks/visibility.
            if len(self.track_out_keys) != 1 or len(self.visibility_out_keys) != 1:
                raise ValueError(
                    f"{self.track_mode} requires exactly one track_out_key and one visibility_out_key. "
                    f"Got {len(self.track_out_keys)=}, {len(self.visibility_out_keys)=}."
                )

        self.rgb_key = rgb_key

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    @staticmethod
    def _normalize_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
        """
        Normalize various possible mask shapes to (T, H, W).

        Common shapes observed in code/comments:
          - (T, H, W)
          - (T, 1, H, W)
          - (1, T, 1, H, W)
          - (1, T, H, W)
        """
        # Squeeze any singleton dims except we keep time if present.
        # We'll collapse to at most 3 dims by squeezing leading/trailing singleton dims.
        # After squeezing, accept either (T,H,W) or (T,1,H,W) -> squeeze channel.
        while mask.dim() > 3 and mask.size(0) == 1:
            mask = mask.squeeze(0)
        if mask.dim() == 4 and mask.size(1) == 1:
            mask = mask.squeeze(1)
        if mask.dim() != 3:
            raise ValueError(
                f"Unsupported mask shape after normalization: {tuple(mask.shape)}"
            )
        return mask

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        for camera_key in self.camera_keys:
            traj_len = tensordict["action"].shape[0]
            video = (
                tensordict["obs"][camera_key][self.rgb_key].permute(0, 3, 1, 2).float()
            )

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
                grid_indices = torch.cartesian_prod(
                    zeros_tensor, xs, ys
                )  # (M,3) with (0,x,y)

            # Prepare per-output accumulators.
            # In masked_grid mode, we create one accumulator per mask/output key.
            # In other modes, we create a single accumulator.
            if self.track_mode == "masked_grid":
                assert self.mask_keys is not None
                track_lists: List[list] = [[] for _ in self.mask_keys]
                visibility_lists: List[list] = [[] for _ in self.mask_keys]

                # Load all masks once (on device), normalize to (T,H,W)
                masks: List[torch.Tensor] = []
                for mk in self.mask_keys:
                    m = tensordict["obs"][camera_key][mk].to(self.device).float()
                    m = self._normalize_mask_tensor(m)
                    masks.append(m)
            else:
                track_list: list = []
                visibility_list: list = []

            for start_frame in tqdm.tqdm(range(traj_len), desc="Cotracker tracking"):
                sub_video = video[start_frame : start_frame + self.track_len].to(
                    self.device
                )

                # Select queries
                per_mask_union_membership: List[torch.Tensor] | None = None

                if self.track_mode == "grid":
                    local_grid_indices = grid_indices

                elif self.track_mode == "masked_grid":
                    # Evaluate each mask on the full grid, OR them to get union selection
                    assert self.mask_keys is not None
                    assert per_mask_union_membership is None

                    # grid_indices: (M,3) with (frame,x,y); masks indexed as [t, y, x]
                    grid_y = grid_indices[:, 2]
                    grid_x = grid_indices[:, 1]

                    per_mask_full: List[torch.Tensor] = []
                    for m in masks:
                        sub_mask = m[start_frame]  # (H,W) on self.device
                        vals = sub_mask[grid_y, grid_x] > 0.5  # (M,) on self.device
                        per_mask_full.append(vals)

                    union = per_mask_full[0].clone()
                    for vals in per_mask_full[1:]:
                        union |= vals

                    local_grid_indices = grid_indices[union]

                    # IMPORTANT: pred_tracks/pred_visibility are moved to CPU below,
                    # so these membership masks must also be on CPU for indexing.
                    per_mask_union_membership = [
                        vals[union].cpu() for vals in per_mask_full
                    ]

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

                else:
                    raise RuntimeError(f"Unexpected track_mode: {self.track_mode}")

                pred_tracks, pred_visibility = self.cotracker(
                    sub_video[None],
                    grid_size=0,
                    queries=local_grid_indices[None].float(),
                )

                pred_tracks = pred_tracks[0].cpu()  # (T_sub, N, 2)
                pred_visibility = pred_visibility[0].cpu()  # (T_sub, N)

                # Pad predictions if at end of trajectory
                if start_frame + self.track_len > traj_len:
                    pad_size = start_frame + self.track_len - traj_len
                    pred_tracks = torch.cat(
                        [pred_tracks, pred_tracks[-1].repeat(pad_size, 1, 1)], dim=0
                    )
                    pred_visibility = torch.cat(
                        [pred_visibility, pred_visibility[-1].repeat(pad_size, 1)],
                        dim=0,
                    )

                # Move jagged dim first (N, traj_len, 2) and (N, traj_len)
                pred_tracks = pred_tracks.permute(1, 0, 2)
                pred_visibility = pred_visibility.permute(1, 0)

                if self.track_mode == "masked_grid":
                    assert per_mask_union_membership is not None
                    # Split union-tracked points back into per-mask outputs
                    for i, member in enumerate(per_mask_union_membership):
                        # member is boolean over union-selected points (shape: N_union)
                        track_lists[i].append(pred_tracks[member])
                        visibility_lists[i].append(pred_visibility[member])
                else:
                    track_list.append(pred_tracks)
                    visibility_list.append(pred_visibility)

            # Write outputs
            if self.track_mode == "masked_grid":
                for out_k, vis_k, tl, vl in zip(
                    self.track_out_keys,
                    self.visibility_out_keys,
                    track_lists,
                    visibility_lists,
                ):
                    # fmt: off
                    tensordict["obs"][camera_key][out_k] = torch.nested.as_nested_tensor(tl, layout=torch.jagged)
                    tensordict["obs"][camera_key][vis_k] = torch.nested.as_nested_tensor(vl, layout=torch.jagged)
                    # fmt: on
            else:
                # fmt: off
                tensordict["obs"][camera_key][self.track_out_keys[0]] = torch.nested.as_nested_tensor(track_list, layout=torch.jagged)
                tensordict["obs"][camera_key][self.visibility_out_keys[0]] = torch.nested.as_nested_tensor(visibility_list, layout=torch.jagged)
                # fmt: on

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Do nothing if not called during preprocessing
        return tensordict
