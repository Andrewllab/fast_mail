from __future__ import annotations

import logging
from typing import Dict, List, Literal, Sequence

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from environments.specs import (
    CameraSpec,
    DataSpecs,
    DepthStream,
    NestedTensorSpec,
    PointCloudSpec,
)
from transforms.base_transform import Transform
from utils.nested import nested_safe_tensordict_unsqueeze

log = logging.getLogger(__name__)


class SparseToPointCloudMerged(Transform):
    """Sparse -> pointcloud transform supporting two modes.

    Modes:
        - "mask": sample one 3D point per masked patch (MaskOnly behavior)
        - "track": unproject tracked 2D keypoints over a short horizon (TrackOnly behavior)

    The output is written to tensordict["obs", out_key] as a dict of nested (jagged)
    tensors, matching the behavior of the original two transforms.
    """

    FEATURE_DIM: int = 768  # TODO: infer from specs instead of hard-coding
    PATCH_SIZE: int = 16  # TODO: infer from specs instead of hard-coding

    def __init__(
        self,
        specs: DataSpecs,
        mode: Literal["mask", "track"] = "mask",
        color: bool = False,
        features: bool = False,
        camera_keys: Sequence[str] | None = None,
        # mask mode
        mask_key: str | Sequence[str] | None = None,
        # track mode
        track_key: str | Sequence[str] | None = None,
        visibility_key: str | Sequence[str] | None = None,
        # shared
        features_key: str | Sequence[str] | None = None,
        max_depth: float | None = None,
        out_key: str = "pcd",
    ) -> None:
        if camera_keys is None:
            raise ValueError("camera_keys must be provided")
        if mode not in ("mask", "track"):
            raise ValueError(f"Unsupported mode={mode!r}; expected 'mask' or 'track'")

        self.mode: Literal["mask", "track"] = mode
        self.color = color
        self.features = features
        self.max_depth = max_depth
        self._out_key = out_key

        self.camera_keys = list(camera_keys)
        self.mask_key = mask_key
        self.track_key = track_key
        self.visibility_key = visibility_key
        self.features_key = features_key

        if self.mode == "mask" and self.mask_key is None:
            raise ValueError("mask_key must be provided when mode='mask'")
        if self.mode == "track":
            if self.track_key is None:
                raise ValueError("track_key must be provided when mode='track'")
            if self.visibility_key is None:
                raise ValueError("visibility_key must be provided when mode='track'")
            self.track_len = specs.action_seq_len + 1

        if self.features_key is None:
            raise ValueError("features_key must be provided")

        # Copy obs specs for local modification
        obs_specs = dict(specs.obs)
        obs_specs[self._out_key] = NestedTensorSpec(time=False)
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    # --------------------------- helpers ---------------------------
    def _get_cam_spec(self, camera_key: str) -> CameraSpec:
        cam_spec = self._output_specs.obs[camera_key]
        if not isinstance(cam_spec, CameraSpec):
            raise ValueError(f"obs[{camera_key}] is not a CameraSpec")
        return cam_spec

    @staticmethod
    def _get_depth_stream(cam_spec: CameraSpec, camera_key: str) -> DepthStream:
        depth_stream: DepthStream | None = None
        for stream in cam_spec.streams.values():
            if isinstance(stream, DepthStream):
                depth_stream = stream
                break
        if depth_stream is None or depth_stream.intrinsics is None:
            raise ValueError(
                f"No DepthStream with intrinsics found for camera {camera_key}"
            )
        return depth_stream

    def _get_intrinsics(
        self, depths: torch.Tensor, cam_spec: CameraSpec, camera_key: str
    ) -> tuple[torch.Tensor, bool]:
        depth_stream = self._get_depth_stream(cam_spec, camera_key)
        K = depth_stream.intrinsics.intrinsic_matrix.to(
            device=depths.device, dtype=torch.float32
        )
        is_ortho = bool(depth_stream.orthogonal)
        return K, is_ortho

    def _get_transforms(
        self, tensordict: TensorDict, cam_spec: CameraSpec, *, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return (T_static, dynamic_T).

        - T_static: (4,4) constant extrinsics
        - dynamic_T: (T,4,4) or (B,4,4) dynamic transform pulled from obs, if present
        """
        T_static = None
        if cam_spec.extrinsics is not None:
            T_static = cam_spec.extrinsics.to(device=device, dtype=torch.float32)

        dynamic_T = None
        pose_key = cam_spec.dynamic_pose_obs_key
        if pose_key is not None:
            if not isinstance(pose_key, tuple):
                pose_key = (pose_key,)
            dynamic_T = tensordict[("obs",) + pose_key].to(
                device=device, dtype=torch.float32
            )

        return T_static, dynamic_T

    @staticmethod
    def _filter_depth(d: torch.Tensor, *, max_depth: float | None) -> torch.Tensor:
        valid = d > 0
        if max_depth is not None:
            valid = valid & (d < float(max_depth))
        return valid

    @staticmethod
    def _apply_extrinsics(
        pts: torch.Tensor,  # (...,3)
        *,
        T_static: torch.Tensor | None,
        dynamic_T: torch.Tensor | None,
        dynamic_index: torch.Tensor | None,  # (...) indexes into dynamic_T
    ) -> torch.Tensor:
        """Apply static and/or dynamic extrinsics to points.

        Works with arbitrary leading dimensions; if dynamic_T is used, dynamic_index
        must broadcast to the leading shape of pts.
        """
        if T_static is None and dynamic_T is None:
            return pts

        orig_shape = pts.shape
        pts_f = pts.reshape(-1, 3)

        if dynamic_T is not None:
            if dynamic_index is None:
                raise ValueError(
                    "dynamic_index must be provided when dynamic_T is used"
                )
            idx_f = dynamic_index.reshape(-1).to(dtype=torch.long)
            T_eff = dynamic_T[idx_f]
            if T_static is not None:
                T_eff = T_eff @ T_static
            R = T_eff[:, :3, :3]
            t = T_eff[:, :3, 3]
            out = torch.einsum("nij,nj->ni", R, pts_f) + t
            return out.reshape(orig_shape)

        # Only static
        R = T_static[:3, :3]
        t = T_static[:3, 3]
        return ((pts_f @ R.T) + t).reshape(orig_shape)

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        if self.mode == "mask":
            return self._call_mask(tensordict)
        return self._call_track(tensordict)

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        return nested_safe_tensordict_unsqueeze(
            self.call_trajectory(tensordict[0]), dim=0
        )

    # --------------------------- mask mode ---------------------------

    def _call_mask(self, tensordict: TensorDict) -> TensorDict:
        mask_key = self.mask_key
        features_key = self.features_key

        # Batch size is the leading dimension for camera observations
        B = int(tensordict["obs", self.camera_keys[0], "depth"].shape[0])

        acc_points: list[list[torch.Tensor]] = [[] for _ in range(B)]
        acc_colors: list[list[torch.Tensor]] = [[] for _ in range(B)]
        acc_features: list[list[torch.Tensor]] = [[] for _ in range(B)]

        for camera_key in self.camera_keys:
            rgbs = tensordict["obs", camera_key, "rgb"]
            depths = tensordict["obs", camera_key, "depth"]
            masks = tensordict["obs", camera_key, mask_key]
            features = tensordict["obs", camera_key, features_key]

            cam_spec = self._get_cam_spec(camera_key)
            K, is_ortho = self._get_intrinsics(depths, cam_spec, camera_key)
            T_static, dynamic_T = self._get_transforms(
                tensordict, cam_spec, device=depths.device
            )

            # Features define the patch grid
            Hp, Wp = int(features.shape[1]), int(features.shape[2])
            patch = self.PATCH_SIZE
            cy_off = patch // 2
            cx_off = patch // 2

            Hc = Hp * patch
            Wc = Wp * patch

            # Crop image-like tensors to patch-aligned area
            depths_c = depths[:, :Hc, :Wc]
            masks_c = masks[:, :Hc, :Wc]
            rgbs_c = rgbs[:, :Hc, :Wc, :]

            # Keep patch if any pixel in its patch window is masked
            patch_mask = _pool_mask_to_patches(
                masks_c, Hp=Hp, Wp=Wp, patch=patch
            )  # (B,Hp,Wp)

            b_idx, py, px = patch_mask.nonzero(as_tuple=True)
            if b_idx.numel() == 0:
                continue

            # Center pixel coords per selected patch
            v = py * patch + cy_off
            u = px * patch + cx_off

            # Depth at centers
            d = depths_c[b_idx, v, u].to(dtype=torch.float32)

            valid = self._filter_depth(d, max_depth=self.max_depth)
            if not valid.any():
                continue

            b_idx = b_idx[valid]
            py = py[valid]
            px = px[valid]
            v = v[valid]
            u = u[valid]
            d = d[valid]

            pts = _sparse_unproject(u, v, d, K, is_ortho=is_ortho)  # (N,3)

            # Apply extrinsics (static + optional dynamic indexed by batch element)
            pts = self._apply_extrinsics(
                pts, T_static=T_static, dynamic_T=dynamic_T, dynamic_index=b_idx
            )

            cols = rgbs_c[b_idx, v, u].to(dtype=torch.float32)  # (N,3)
            feats = features[b_idx, py, px].to(dtype=torch.float32)  # (N,FEATURE_DIM)

            # Scatter into per-batch ragged buffers
            for b in b_idx.unique().tolist():
                sel = b_idx == b
                acc_points[b].append(pts[sel])
                acc_colors[b].append(cols[sel])
                acc_features[b].append(feats[sel])

        data_list: List[Dict] = []
        for b in range(B):
            if len(acc_points[b]) == 0:
                pts_b = torch.empty((0, 3), device=depths.device, dtype=torch.float32)
                cols_b = torch.empty((0, 3), device=depths.device, dtype=torch.float32)
                feats_b = torch.empty(
                    (0, self.FEATURE_DIM), device=depths.device, dtype=torch.float32
                )
            else:
                pts_b = torch.cat(acc_points[b], dim=0)
                cols_b = torch.cat(acc_colors[b], dim=0)
                feats_b = torch.cat(acc_features[b], dim=0)

            data_list.append({"points": pts_b, "features": feats_b, "colors": cols_b})

        # Assemble ragged tensors
        out_dict = {
            "points": torch.nested.as_nested_tensor(
                [data["points"] for data in data_list], layout=torch.jagged
            ),
            "features": torch.nested.as_nested_tensor(
                [data["features"] for data in data_list], layout=torch.jagged
            ),
            "colors": torch.nested.as_nested_tensor(
                [data["colors"] for data in data_list], layout=torch.jagged
            ),
        }
        tensordict["obs", self._out_key] = out_dict
        return tensordict

    # --------------------------- track mode ---------------------------

    def _call_track(self, tensordict: TensorDict) -> TensorDict:
        track_key = self.track_key
        visibility_key = self.visibility_key
        features_key = self.features_key

        video_len = int(tensordict["obs", self.camera_keys[0], "depth"].shape[0])

        acc_points: list[list[torch.Tensor]] = [[] for _ in range(video_len)]
        acc_visibility: list[list[torch.Tensor]] = [[] for _ in range(video_len)]
        acc_colors: list[list[torch.Tensor]] = [[] for _ in range(video_len)]
        acc_features: list[list[torch.Tensor]] = [[] for _ in range(video_len)]

        for camera_key in self.camera_keys:
            tracks = tensordict["obs", camera_key, track_key]
            visibility = tensordict["obs", camera_key, visibility_key]
            rgbs = tensordict["obs", camera_key, "rgb"]
            depths = tensordict["obs", camera_key, "depth"]
            features = tensordict["obs", camera_key, features_key]

            cam_spec = self._get_cam_spec(camera_key)
            K, is_ortho = self._get_intrinsics(depths, cam_spec, camera_key)
            T_static, dynamic_T = self._get_transforms(
                tensordict, cam_spec, device=depths.device
            )

            for start_idx in range(video_len):
                sub_tracks = tracks[start_idx]  # (J, L, 2)
                sub_visibility = visibility[start_idx]  # (J, L)
                sub_rgb = rgbs[start_idx]  # (H, W, 3)
                sub_features = features[start_idx]  # (Hp, Wp, FEATURE_DIM)
                sub_depths = depths[
                    start_idx : start_idx + self.track_len
                ]  # (L_eff, H, W)

                L_eff = int(sub_depths.shape[0])
                if L_eff <= 0:
                    continue

                J = int(sub_tracks.shape[0])
                H, W = int(sub_depths.shape[1]), int(sub_depths.shape[2])

                tracks_xy = sub_tracks[:, :L_eff, :]  # (J, L_eff, 2)
                vis = sub_visibility[:, :L_eff].bool()  # (J, L_eff)

                u = torch.round(tracks_xy[..., 1]).long()  # (J, L_eff)
                v = torch.round(tracks_xy[..., 0]).long()  # (J, L_eff)

                in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
                vis = vis & in_bounds

                u_cl = u.clamp(0, W - 1)
                v_cl = v.clamp(0, H - 1)

                t_idx = (
                    torch.arange(L_eff, device=sub_depths.device)
                    .view(1, L_eff)
                    .expand(J, L_eff)
                )
                d = sub_depths[t_idx, v_cl, u_cl].to(dtype=torch.float32)  # (J, L_eff)

                vis = vis & self._filter_depth(d, max_depth=self.max_depth)

                # Points in camera frame, invalid entries are NaN (keep shape)
                points_cam = torch.full(
                    (J, L_eff, 3),
                    float("nan"),
                    device=sub_depths.device,
                    dtype=torch.float32,
                )

                flat_valid = vis.reshape(-1)
                if flat_valid.any():
                    u_flat = u_cl.reshape(-1)[flat_valid]
                    v_flat = v_cl.reshape(-1)[flat_valid]
                    d_flat = d.reshape(-1)[flat_valid]
                    pts_valid = _sparse_unproject(
                        u_flat, v_flat, d_flat, K, is_ortho=is_ortho
                    )
                    points_cam.reshape(-1, 3)[flat_valid] = pts_valid

                if dynamic_T is not None:
                    L_eff = int(points_cam.shape[1])
                    t_idx = (start_idx + torch.arange(L_eff, device=points_cam.device))[
                        None, :
                    ].expand(points_cam.shape[0], L_eff)
                else:
                    t_idx = None

                points_world = self._apply_extrinsics(
                    points_cam,
                    T_static=T_static,
                    dynamic_T=dynamic_T,
                    dynamic_index=t_idx,
                )

                # Color / feature from the first timestep (t=0) track location
                u0 = u_cl[:, 0]
                v0 = v_cl[:, 0]

                cols0 = sub_rgb[v0, u0].to(dtype=torch.float32)  # (J,3)

                Hp, Wp = int(sub_features.shape[0]), int(sub_features.shape[1])
                patch = self.PATCH_SIZE
                py0 = (v0 // patch).clamp(0, Hp - 1)
                px0 = (u0 // patch).clamp(0, Wp - 1)
                feats0 = sub_features[py0, px0].to(
                    dtype=torch.float32
                )  # (J,FEATURE_DIM)

                # Pad to fixed length track_len if needed
                if L_eff < self.track_len:
                    pad_pts = torch.full(
                        (J, self.track_len - L_eff, 3),
                        float("nan"),
                        device=points_world.device,
                        dtype=torch.float32,
                    )
                    pad_vis = torch.zeros(
                        (J, self.track_len - L_eff),
                        device=vis.device,
                        dtype=torch.bool,
                    )
                    points_world = torch.cat([points_world, pad_pts], dim=1)
                    vis = torch.cat([vis, pad_vis], dim=1)

                if (~vis[:, 0]).any() or points_world[:, 0].isnan().any():
                    log.warning(
                        "Some tracks are not visible at the first timestep; these will have invalid points."
                    )

                acc_points[start_idx].append(points_world)  # (J,L,3)
                acc_visibility[start_idx].append(vis)  # (J,L)
                acc_colors[start_idx].append(cols0)  # (J,3)
                acc_features[start_idx].append(feats0)  # (J,FEATURE_DIM)

        data_list: List[Dict] = []
        for start_idx in range(video_len):
            if len(acc_points[start_idx]) == 0:
                pts = torch.empty(
                    (0, self.track_len, 3),
                    device=depths.device,
                    dtype=torch.float32,
                )
                vis = torch.empty(
                    (0, self.track_len), device=depths.device, dtype=torch.bool
                )
                cols = torch.empty((0, 3), device=depths.device, dtype=torch.float32)
                feats = torch.empty(
                    (0, self.FEATURE_DIM), device=depths.device, dtype=torch.float32
                )
            else:
                pts = torch.cat(acc_points[start_idx], dim=0)
                vis = torch.cat(acc_visibility[start_idx], dim=0)
                cols = torch.cat(acc_colors[start_idx], dim=0)
                feats = torch.cat(acc_features[start_idx], dim=0)

            data_list.append(
                {"points": pts, "visibility": vis, "colors": cols, "features": feats}
            )

        # Assemble ragged tensors
        out_dict = {
            "points": torch.nested.as_nested_tensor(
                [data["points"] for data in data_list], layout=torch.jagged
            ),
            "visibility": torch.nested.as_nested_tensor(
                [data["visibility"] for data in data_list], layout=torch.jagged
            ),
            "colors": torch.nested.as_nested_tensor(
                [data["colors"] for data in data_list], layout=torch.jagged
            ),
            "features": torch.nested.as_nested_tensor(
                [data["features"] for data in data_list], layout=torch.jagged
            ),
        }

        tensordict["obs", self._out_key] = out_dict
        return tensordict


class SparseToPointCloudMaskOnly(SparseToPointCloudMerged):
    """Sparse -> pointcloud transform using only mask mode.

    See SparseToPointCloudMerged for details.
    """

    def __init__(
        self,
        specs: DataSpecs,
        *,
        color: bool = False,
        features: bool = False,
        camera_keys: Sequence[str] | None = None,
        mask_key: str | Sequence[str] | None = None,
        features_key: str | Sequence[str] | None = None,
        max_depth: float | None = None,
        out_key: str = "pcd",
    ) -> None:
        super().__init__(
            specs,
            mode="mask",
            color=color,
            features=features,
            camera_keys=camera_keys,
            mask_key=mask_key,
            features_key=features_key,
            max_depth=max_depth,
            out_key=out_key,
        )


class SparseToPointCloudTrackOnly(SparseToPointCloudMerged):
    """Sparse -> pointcloud transform using only track mode.

    See SparseToPointCloudMerged for details.
    """

    def __init__(
        self,
        specs: DataSpecs,
        *,
        color: bool = False,
        features: bool = False,
        camera_keys: Sequence[str] | None = None,
        track_key: str | Sequence[str] | None = None,
        visibility_key: str | Sequence[str] | None = None,
        features_key: str | Sequence[str] | None = None,
        max_depth: float | None = None,
        out_key: str = "pcd",
    ) -> None:
        super().__init__(
            specs,
            mode="track",
            color=color,
            features=features,
            camera_keys=camera_keys,
            track_key=track_key,
            visibility_key=visibility_key,
            features_key=features_key,
            max_depth=max_depth,
            out_key=out_key,
        )


def _sparse_unproject(
    u: torch.Tensor,  # (N,) pixel x
    v: torch.Tensor,  # (N,) pixel y
    d: torch.Tensor,  # (N,) depth
    K: torch.Tensor,  # (3,3)
    *,
    is_ortho: bool = True,
) -> torch.Tensor:
    """Sparse equivalent of utils.math.unproject_depth for selected pixels only.

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
        d = d / torch.sqrt(1.0 + x_over_z**2 + y_over_z**2)

    x = x_over_z * d
    y = y_over_z * d
    z = d
    return torch.stack([x, y, z], dim=-1)


def _pool_mask_to_patches(
    mask_hw: torch.Tensor, Hp: int, Wp: int, patch: int = 16
) -> torch.Tensor:
    """Downsample a pixel mask to a patch grid using max-pooling.

    Args:
        mask_hw: (B,H,W) bool/0-1
        Hp, Wp: patch grid size
        patch: patch size in pixels

    Returns:
        (B,Hp,Wp) bool patch mask
    """
    Hc = Hp * patch
    Wc = Wp * patch
    mask_crop = mask_hw[:, :Hc, :Wc]

    m = mask_crop.to(dtype=torch.float32).unsqueeze(1)  # (B,1,Hc,Wc)
    pooled = F.max_pool2d(m, kernel_size=patch, stride=patch)  # (B,1,Hp,Wp)
    return pooled.squeeze(1) > 0.0
