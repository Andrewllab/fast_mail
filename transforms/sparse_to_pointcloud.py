from __future__ import annotations

import itertools
import logging

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor
from torch_geometric.data import Batch, Data

from environments.datamodule import EmptyPointCloudError
from environments.specs import (
    CameraSpec,
    DataSpecs,
    DepthStream,
    PointCloudSpec,
    RGBStream,
)
from transforms.base_transform import Transform
from utils.math import transform_pointmap, unproject_depth

log = logging.getLogger(__name__)


class SparseToPointCloudMaskOnly(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        color: bool = False,
        features: bool = False,
        camera_keys: list[str] | None = None,
        mask_key: list[str] | None = None,
        features_key: list[str] | None = None,
        max_depth: float | None = None,
        out_key: str = "pcd",
    ):
        self.color = color
        self.max_depth = max_depth
        self._out_key = out_key
        self.camera_keys = camera_keys
        self.mask_key = mask_key
        self.features = features
        self.features_key = features_key
        if isinstance(self.camera_keys, str):
            self.camera_keys = [self.camera_keys]

        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        obs_specs[self._out_key] = PointCloudSpec(
            feature_dim=768, color=color
        )  # TODO: Make feature dim read from input specs
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # collect points, masks, and rgb from all cameras
        all_points = []
        all_masks = []  # if self.max_depth is not None, populate with masks
        all_rgb = []  # if self.color is True, populate with rgb

        for camera_key in self.camera_keys:
            rgbs = tensordict["obs", camera_key, "rgb"]
            depths = tensordict["obs", camera_key, "depth"]
            masks = tensordict["obs", camera_key, self.mask_key]
            features = tensordict["obs", camera_key, self.features_key]

            pass
            # ADD LOGIC HERE

            # --- inside for camera_key in self.camera_keys: ---
            cam_spec = self._output_specs.obs[camera_key]
            if not isinstance(cam_spec, CameraSpec):
                raise ValueError(f"obs[{camera_key}] is not a CameraSpec")

            # Find depth stream to get intrinsics / ortho flag (same checks as ToPointCloud) :contentReference[oaicite:4]{index=4}
            depth_stream = None
            for stream in cam_spec.streams.values():
                if isinstance(stream, DepthStream):
                    depth_stream = stream
                    break
            if depth_stream is None or depth_stream.intrinsics is None:
                raise ValueError(
                    f"No DepthStream with intrinsics found for camera {camera_key}"
                )

            K = depth_stream.intrinsics.intrinsic_matrix.to(
                device=depths.device, dtype=torch.float32
            )
            is_ortho = depth_stream.orthogonal

            # Features define the patch grid
            B = depths.shape[0]
            Hp, Wp = features.shape[1], features.shape[2]
            patch = 16  # TODO: Make inferred from specs
            cy_off = patch // 2
            cx_off = patch // 2

            Hc = Hp * patch
            Wc = Wp * patch

            # Crop all image-like tensors to patch-aligned area
            depths_c = depths[:, :Hc, :Wc]
            masks_c = masks[:, :Hc, :Wc]
            rgbs_c = rgbs[:, :Hc, :Wc, :]

            # Patch mask: keep patch if *any* pixel in its 16x16 is masked
            patch_mask = _pool_mask_to_patches(
                masks_c, Hp=Hp, Wp=Wp, patch=patch
            )  # (B,Hp,Wp)

            # Indices of selected patches
            b_idx, py, px = patch_mask.nonzero(as_tuple=True)
            if b_idx.numel() == 0:
                # no points from this camera
                continue

            # Center pixel coords for each selected patch
            v = py * patch + cy_off
            u = px * patch + cx_off

            # Gather depth at centers
            d = depths_c[b_idx, v, u].to(dtype=torch.float32)

            # Depth validity / max depth filtering (same spirit as ToPointCloud) :contentReference[oaicite:5]{index=5}
            valid = d > 0
            if self.max_depth is not None:
                valid = valid & (d < float(self.max_depth))

            if not valid.any():
                continue

            b_idx = b_idx[valid]
            py = py[valid]
            px = px[valid]
            v = v[valid]
            u = u[valid]
            d = d[valid]

            # Sparse unprojection (camera frame) :contentReference[oaicite:6]{index=6}
            pts = _sparse_unproject(u, v, d, K, is_ortho=is_ortho)  # (N,3)

            # Apply (static + optional dynamic) extrinsics like ToPointCloud :contentReference[oaicite:7]{index=7}
            if (T := cam_spec.extrinsics) is not None:
                T = T.to(device=pts.device, dtype=torch.float32)

                if (pose_key := cam_spec.dynamic_pose_obs_key) is not None:
                    if not isinstance(pose_key, tuple):
                        pose_key = (pose_key,)
                    dynamic_T = tensordict[("obs",) + pose_key].to(
                        device=pts.device, dtype=torch.float32
                    )
                    # dynamic is left-multiplied (same as ToPointCloud) :contentReference[oaicite:8]{index=8}
                    # Here we assume per-frame (B,4,4) and index with b_idx
                    T_eff = dynamic_T[b_idx] @ T
                else:
                    # broadcast T per selected point
                    T_eff = T.expand(b_idx.shape[0], 4, 4)

                # Apply per-point
                R = T_eff[:, :3, :3]
                t = T_eff[:, :3, 3]
                pts = torch.bmm(R, pts.unsqueeze(-1)).squeeze(-1) + t

            # Gather per-point color + DINO feature
            cols = rgbs_c[b_idx, v, u].to(dtype=torch.float32)  # (N,3)
            feats = features[b_idx, py, px].to(dtype=torch.float32)  # (N,768)

            # Store per-camera contributions; we’ll concatenate across cameras later
            all_points.append((b_idx, pts))
            all_rgb.append((b_idx, cols))
            all_masks.append(
                (b_idx, feats)
            )  # reusing list name as "all_feats" container

            B = tensordict["obs", self.camera_keys[0], "depth"].shape[0]

        # Prepare ragged outputs: list of length B, each element is a dict
        out = [{"points": [], "features": [], "colors": []} for _ in range(B)]

        for (b_idx, pts), (_, cols), (_, feats) in zip(all_points, all_rgb, all_masks):
            # group by batch index (masks are sparse, looping is fine)
            for b in b_idx.unique().tolist():
                sel = b_idx == b
                out[b]["points"].append(pts[sel])
                out[b]["colors"].append(cols[sel])
                out[b]["features"].append(feats[sel])

        # Final concat per batch item
        final = []
        for b in range(B):
            if len(out[b]["points"]) == 0:
                # keep empty (downstream can decide what to do)
                pts_b = torch.empty((0, 3), device=depths.device, dtype=torch.float32)
                cols_b = torch.empty((0, 3), device=depths.device, dtype=torch.float32)
                feats_b = torch.empty(
                    (0, 768), device=depths.device, dtype=torch.float32
                )
            else:
                pts_b = torch.cat(out[b]["points"], dim=0)
                cols_b = torch.cat(out[b]["colors"], dim=0)
                feats_b = torch.cat(out[b]["features"], dim=0)

            final.append({"points": pts_b, "features": feats_b, "colors": cols_b})

        tensordict["obs", self._out_key] = Batch.from_data_list(
            [
                Data(
                    pos=final[i]["points"],
                    x=final[i]["features"],
                    color=final[i]["colors"],
                )
                for i in range(B)
            ]
        )
        return tensordict

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self(tensordict)
        return tensordict


class SparseToPointCloudTrackOnly(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        color: bool = False,
        features: bool = False,
        camera_keys: list[str] | None = None,
        track_key: list[str] | None = None,
        visibility_key: list[str] | None = None,
        features_key: list[str] | None = None,
        max_depth: float | None = None,
        out_key: str = "pcd",
    ):
        self.color = color
        self.max_depth = max_depth
        self._out_key = out_key
        self.camera_keys = camera_keys
        self.track_key = track_key
        self.visibility_key = visibility_key
        self.features = features
        self.features_key = features_key
        if isinstance(self.camera_keys, str):
            self.camera_keys = [self.camera_keys]

        self.action_seq_len = specs.action_seq_len

        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        obs_specs[self._out_key] = PointCloudSpec(
            feature_dim=768, color=color
        )  # TODO: Make feature dim read from input specs
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # collect points, masks, and rgb from all cameras
        video_len = tensordict["obs", self.camera_keys[0], "depth"].shape[0]
        acc_points = [[] for _ in range(video_len)]
        acc_visibility = [[] for _ in range(video_len)]
        acc_rgb = [[] for _ in range(video_len)]
        acc_features = [[] for _ in range(video_len)]

        for camera_key in self.camera_keys:
            tracks = tensordict["obs", camera_key, self.track_key]
            visibility = tensordict["obs", camera_key, self.visibility_key]
            rgbs = tensordict["obs", camera_key, "rgb"]
            depths = tensordict["obs", camera_key, "depth"]
            features = tensordict["obs", camera_key, self.features_key]

            video_len = depths.shape[0]  # trajectory length

            cam_spec = self._output_specs.obs[camera_key]
            if not isinstance(cam_spec, CameraSpec):
                raise ValueError(f"obs[{camera_key}] is not a CameraSpec")

            depth_stream = None
            for stream in cam_spec.streams.values():
                if isinstance(stream, DepthStream):
                    depth_stream = stream
                    break
            if depth_stream is None or depth_stream.intrinsics is None:
                raise ValueError(
                    f"No DepthStream with intrinsics found for camera {camera_key}"
                )

            K = depth_stream.intrinsics.intrinsic_matrix.to(
                device=depths.device, dtype=torch.float32
            )
            is_ortho = depth_stream.orthogonal

            T_static = None
            if cam_spec.extrinsics is not None:
                T_static = cam_spec.extrinsics.to(
                    device=depths.device, dtype=torch.float32
                )

            pose_key = cam_spec.dynamic_pose_obs_key
            dynamic_T = None
            if pose_key is not None:
                if not isinstance(pose_key, tuple):
                    pose_key = (pose_key,)
                dynamic_T = tensordict[("obs",) + pose_key].to(
                    device=depths.device, dtype=torch.float32
                )  # (video_len,4,4) typically

            for start_idx in range(video_len):
                # Since we tracked over action_seq_len frames, we need to apply the sparse unprojection per sub-sequence of length action_seq_len
                sub_tracks = tracks[start_idx]
                sub_visibility = visibility[start_idx]
                sub_rgbs = rgbs[start_idx]
                sub_features = features[start_idx]
                sub_depths = depths[start_idx : start_idx + self.action_seq_len]

                # sub_tracks: (J, L, 2), sub_visibility: (J, L)
                # sub_rgbs: (H, W, 3), sub_depths: (L, H, W)
                J = sub_tracks.shape[0]
                L = self.action_seq_len
                H, W = sub_depths.shape[1], sub_depths.shape[2]

                # Handle tail: if not enough frames left, truncate consistently
                L_eff = sub_depths.shape[0]
                if L_eff <= 0:
                    continue

                tracks_xy = sub_tracks[:, :L_eff, :]  # (J, L_eff, 2)
                vis = sub_visibility[:, :L_eff].bool()  # (J, L_eff)

                # Pixel coords (nearest neighbor)
                u = torch.round(tracks_xy[..., 0]).long()  # (J, L_eff)
                v = torch.round(tracks_xy[..., 1]).long()  # (J, L_eff)

                # Bounds check
                in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                vis = vis & in_bounds  # only visible if in bounds too

                # Clamp for safe indexing (values outside will be masked out by vis anyway)
                u_cl = u.clamp(0, W - 1)
                v_cl = v.clamp(0, H - 1)

                # Gather depth at each timestep/pixel (vectorized)
                t_idx = (
                    torch.arange(L_eff, device=sub_depths.device)
                    .view(1, L_eff)
                    .expand(J, L_eff)
                )  # (J, L_eff)
                d = sub_depths[t_idx, v_cl, u_cl].to(dtype=torch.float32)  # (J, L_eff)

                # Depth validity
                valid_d = d > 0
                if self.max_depth is not None:
                    valid_d = valid_d & (d < float(self.max_depth))

                vis = vis & valid_d  # final visibility used for output

                # Allocate points (camera frame), fill invalid with NaN
                points_cam = torch.full(
                    (J, L_eff, 3),
                    float("nan"),
                    device=sub_depths.device,
                    dtype=torch.float32,
                )

                # Sparse unproject only valid entries
                flat_valid = vis.view(-1)
                if flat_valid.any():
                    u_flat = u_cl.view(-1)[flat_valid]
                    v_flat = v_cl.view(-1)[flat_valid]
                    d_flat = d.view(-1)[flat_valid]

                    pts_valid = _sparse_unproject(
                        u_flat, v_flat, d_flat, K, is_ortho=is_ortho
                    )  # (N,3)
                    points_cam.view(-1, 3)[flat_valid] = pts_valid

                # Apply extrinsics (static + dynamic per frame)
                points_world = points_cam
                if T_static is not None or dynamic_T is not None:
                    # copy to avoid in-place surprises
                    points_world = points_cam.clone()

                    for tt in range(L_eff):
                        pts_t = points_world[:, tt, :]  # (J,3)
                        vis_t = vis[:, tt]  # (J,)

                        if not vis_t.any():
                            continue

                        # pick transform for this absolute frame
                        if dynamic_T is not None and T_static is not None:
                            T_eff = dynamic_T[start_idx + tt] @ T_static
                        elif dynamic_T is not None:
                            T_eff = dynamic_T[start_idx + tt]
                        else:
                            T_eff = T_static

                        R = T_eff[:3, :3]
                        t = T_eff[:3, 3]
                        pts_t_vis = pts_t[vis_t]
                        pts_t[vis_t] = (pts_t_vis @ R.T) + t
                        points_world[:, tt, :] = pts_t

                # Initial colors + features from timestep 0 (use the first track position)
                u0 = u_cl[:, 0]
                v0 = v_cl[:, 0]

                # Initial color (J,3)
                cols0 = sub_rgbs[v0, u0].to(dtype=torch.float32)  # (J,3)

                # Initial DINO feature from patch containing (u0,v0)
                Hp, Wp = sub_features.shape[0], sub_features.shape[1]
                patch = 16
                py0 = (v0 // patch).clamp(0, Hp - 1)
                px0 = (u0 // patch).clamp(0, Wp - 1)
                feats0 = sub_features[py0, px0].to(dtype=torch.float32)  # (J,768)

                # If you want: optionally mask out colors/features for points that are never visible
                # (keeping them as-is is often fine; visibility carries the info)
                # never_vis = ~vis.any(dim=1)
                # cols0[never_vis] = 0
                # feats0[never_vis] = 0

                # Pad visibility/points back to full L if you require fixed length
                if L_eff < L:
                    pad_pts = torch.full(
                        (J, L - L_eff, 3),
                        float("nan"),
                        device=points_world.device,
                        dtype=torch.float32,
                    )
                    pad_vis = torch.zeros(
                        (J, L - L_eff), device=vis.device, dtype=torch.bool
                    )
                    points_world = torch.cat([points_world, pad_pts], dim=1)  # (J,L,3)
                    vis = torch.cat([vis, pad_vis], dim=1)  # (J,L)

                # Accumulate for this start_idx (concat across cameras later)
                acc_points[start_idx].append(points_world)  # (J,L,3)
                acc_visibility[start_idx].append(vis)  # (J,L)
                acc_rgb[start_idx].append(cols0)  # (J,3)
                acc_features[start_idx].append(feats0)  # (J,768)

        final = []
        for start_idx in range(video_len):
            if len(acc_points[start_idx]) == 0:
                # No tracks from any camera for this start_idx
                pts = torch.empty(
                    (0, self.action_seq_len, 3),
                    device=depths.device,
                    dtype=torch.float32,
                )
                vis = torch.empty(
                    (0, self.action_seq_len), device=depths.device, dtype=torch.bool
                )
                cols = torch.empty((0, 3), device=depths.device, dtype=torch.float32)
                feats = torch.empty((0, 768), device=depths.device, dtype=torch.float32)
            else:
                pts = torch.cat(
                    acc_points[start_idx], dim=0
                )  # (P,L,3) where P=sum_cameras J
                vis = torch.cat(acc_visibility[start_idx], dim=0)  # (P,L)
                cols = torch.cat(acc_rgb[start_idx], dim=0)  # (P,3)
                feats = torch.cat(acc_features[start_idx], dim=0)  # (P,768)

            final.append(
                {
                    "points": pts,
                    "visibility": vis,
                    "features": feats,
                    "colors": cols,
                }
            )

        tensordict["obs", self._out_key] = Batch.from_data_list(
            [
                Data(
                    pos=final[i]["points"],
                    x=final[i]["features"],
                    color=final[i]["colors"],
                    visibility=final[i]["visibility"],
                )
                for i in range(video_len)
            ]
        )

        return tensordict

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        tensordict = self(tensordict)
        return tensordict


def _sparse_unproject(
    u: torch.Tensor,  # (N,) pixel x
    v: torch.Tensor,  # (N,) pixel y
    d: torch.Tensor,  # (N,) depth
    K: torch.Tensor,  # (3,3)
    is_ortho: bool = True,
) -> torch.Tensor:
    """
    Sparse equivalent of utils.math.unproject_depth for selected pixels only. :contentReference[oaicite:2]{index=2}
    Returns (N,3) points in the camera frame.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    u = u.to(dtype=d.dtype)
    v = v.to(dtype=d.dtype)

    x_over_z = (u - cx) / fx
    y_over_z = (v - cy) / fy

    if not is_ortho:
        # matches orthogonalize_perspective_depth logic but per-pixel :contentReference[oaicite:3]{index=3}
        d = d / torch.sqrt(1.0 + x_over_z**2 + y_over_z**2)

    x = x_over_z * d
    y = y_over_z * d
    z = d
    return torch.stack([x, y, z], dim=-1)


def _apply_extrinsics(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Apply 4x4 extrinsics to (N,3) points.
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return (points @ R.T) + t


def _pool_mask_to_patches(
    mask_hw: torch.Tensor, Hp: int, Wp: int, patch: int = 16
) -> torch.Tensor:
    """
    mask_hw: (B,H,W) bool
    Returns patch_mask: (B,Hp,Wp) bool
    Uses max-pool over each patch (so any True within a patch activates it).
    """
    B, H, W = mask_hw.shape
    Hc = Hp * patch
    Wc = Wp * patch
    mask_crop = mask_hw[:, :Hc, :Wc]

    # max-pool needs float
    m = mask_crop.to(dtype=torch.float32).unsqueeze(1)  # (B,1,Hc,Wc)
    pooled = F.max_pool2d(m, kernel_size=patch, stride=patch)  # (B,1,Hp,Wp)
    return pooled.squeeze(1) > 0.0
