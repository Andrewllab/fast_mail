from __future__ import annotations

import os
from typing import Sequence

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from tensordict import NonTensorData, TensorDict
from torch_geometric.data import Data

from environments.specs import DataSpecs, ObsSpec
from transforms.base_transform import ReversibleTransform, Transform
from utils.math import project_points


class PaktRolloutPointVis(ReversibleTransform):
    def __init__(
        self,
        specs: DataSpecs,
    ) -> None:

        self._output_specs = specs
        self._camera_keys = ["robot0_agentview_left", "robot0_agentview_right"]
        self.output_folder = "/mnt/SmallSandwich/visualization_rollout"
        self.counter = 0

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict):
        return tensordict

    def _get_intrinsics_extrinsics(
        self, camera_key: str, tensordict: TensorDict, device: torch.device
    ) -> tuple[torch.Tensor, bool]:
        cam_spec = self.specs.obs[camera_key]
        # Intrinsics
        intrinsics = cam_spec.intrinsics.intrinsic_matrix.to(device)

        # Static extrinsics
        T_static = None
        if cam_spec.extrinsics is not None:
            T_static = cam_spec.extrinsics.to(device)

            I = torch.eye(4, device=device)
            if torch.allclose(T_static, I, atol=1e-6, rtol=1e-6):
                T_static = None  # treat identity as “no static transform”

        # Optional dynamic extrinsics
        dynamic_T = None
        pose_key = cam_spec.dynamic_pose_obs_key
        if pose_key is not None:
            if not isinstance(pose_key, tuple):
                pose_key = (pose_key,)
            dynamic_T = tensordict[("obs",) + pose_key].to(device)[0]

        return intrinsics, T_static, dynamic_T

    @staticmethod
    def _apply_extrinsics(
        pts: torch.Tensor,  # (...,3)
        *,
        T_static: torch.Tensor | None,
        dynamic_T: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply static and/or dynamic extrinsics to points.

        Works with arbitrary leading dimensions; if dynamic_T is used, dynamic_index
        must broadcast to the leading shape of pts.
        """
        if T_static is None and dynamic_T is None:
            return pts

        if dynamic_T is not None:
            T_eff = dynamic_T
            if T_static is not None:
                T_eff = T_eff @ T_static
            R = T_eff[:3, :3]
            t = T_eff[:3, 3]
            return (pts @ R.T) + t

        # Only static
        R = T_static[:3, :3]
        t = T_static[:3, 3]
        return ((pts @ R.T) + t)

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        action_points = tensordict["action"][0]
        counter = self.counter
        self.counter += 1
        for camera_key in self._camera_keys:
            os.makedirs(os.path.join(self.output_folder, camera_key), exist_ok=True)
            intrinsics, T_static, dynamic_T = self._get_intrinsics_extrinsics(camera_key, tensordict, device=action_points.device)

            # Apply inverse extrinsics to action points to get them back into the camera frame for projection
            if T_static is not None:
                T_static = torch.linalg.inv(T_static)
            if dynamic_T is not None:
                dynamic_T = torch.linalg.inv(dynamic_T)
            pts = self._apply_extrinsics(
                action_points, T_static=T_static, dynamic_T=dynamic_T
            )

            projected_points_ = project_points(pts, intrinsics)
            projected_points = projected_points_[..., :2].reshape(-1, 20, 2)

            robot_points = projected_points[:5]
            obs_points = projected_points[5:]

            # robot_points = robot_points.flip(2)
            # obs_points = obs_points.flip(2)

            rgb_image = tensordict["obs", camera_key, "rgb"][0]
            vis_image = self.draw_tracks_on_frame(
                rgb_image.cpu().numpy(),
                robot_points.cpu().numpy(),
                fixed_color=(0, 0, 255),
                draw_diamond=True,
            )
            vis_image = self.draw_tracks_on_frame(
                vis_image,
                obs_points.cpu().numpy(),
                # fixed_color=(0, 255, 0),
                draw_diamond=False,
            )
            plt.imsave(
                os.path.join(
                    self.output_folder, camera_key, f"{counter:04d}.png"
                ),
                vis_image,
            )

        return tensordict

    @staticmethod
    def draw_tracks_on_frame(frame, tracks, alpha_min=0.2, fixed_color=None, draw_diamond: bool = True):
        # Normalize to uint8 if the frame came in as float (0.0–1.0)
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        # HDF5 frames are RGB; cv2 draws in BGR — convert once at the top
        out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        N, T, _ = tracks.shape

        if fixed_color is not None:
            colors = [fixed_color] * N
        else:
            colors = [
                cv2.cvtColor(np.uint8([[[int(180 * i / N), 220, 220]]]), cv2.COLOR_HSV2BGR)[
                    0, 0
                ].tolist()
                for i in range(N)
            ]

        for n in range(N):
            color = colors[n]
            pts = tracks[n].astype(np.int32)

            for t in range(1, T):
                alpha = alpha_min + (1.0 - alpha_min) * (t / (T - 1))
                pt1, pt2 = tuple(pts[t - 1]), tuple(pts[t])
                seg = out.copy()
                cv2.line(seg, pt1, pt2, color, 2, cv2.LINE_AA)
                cv2.addWeighted(seg, alpha, out, 1 - alpha, 0, out)

            if not draw_diamond:
                start = tuple(pts[0])
                cv2.circle(out, start, 8, (255, 255, 255), -1, cv2.LINE_AA)  # white halo, radius+2
                cv2.circle(out, start, 6, color, -1, cv2.LINE_AA)
                # cv2.circle(out, start, 6, (255, 255, 255), 2, cv2.LINE_AA)

            if draw_diamond:
                ex, ey = pts[0]
                d = 8
                diamond = np.array(
                    [[ex, ey - d], [ex + d, ey], [ex, ey + d], [ex - d, ey]], np.int32
                )
                cv2.fillPoly(out, [diamond], color)
                cv2.polylines(out, [diamond], True, (255, 255, 255), 1, cv2.LINE_AA)

        # Convert back to RGB so plt.imsave renders correctly
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
