from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Data
import open3d as o3d
import numpy as np

from transforms.base_transform import KeyMapping, Transform


if TYPE_CHECKING:
    from torch import Tensor

    from environments.specs import DataSpecs


class CropTablePointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        distance_threshold: float = 0.05, # Max distance a point can be from the plane model, and still be considered an inlier
        ransac_n: int = 5, # Number of initial points to be considered inliers in each iteration
        num_iterations: int = 100, # Number of iterations # TODO: No idea whats a good default value :D
        probability: float = 0.99999999, # Expected probability of finding the optimal plane
    ):
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.probability = probability
        
        self.specs = specs

        self._key_mappings = [
            KeyMapping(
                in_keys=[("obs", "pcd")],
                out_keys=[("obs", "pcd")]
                )]

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return self._key_mappings

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, pc: Data) -> Data:

        # https://archive.ph/XNLup    Beware, this is a medium link.

        pos = pc.pos.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos)

        # color handling
        if hasattr(pc, 'x') and pc.x is not None and pc.x.shape[1] >= 3:
            colors = pc.x[:, :3].detach().cpu().numpy()
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        plane_model, inliers = pcd.segment_plane(
            self.distance_threshold,
            self.ransac_n,
            self.num_iterations,
            self.probability)

        [a, b, c, d] = plane_model

        outlier_mask = np.ones(len(pos), dtype=bool)
        outlier_mask[inliers] = False
        outlier_indices = np.arange(len(pos))[outlier_mask]

        outlier_points = pos[outlier_indices]

        below_mask = (a * outlier_points[:, 0] + b * outlier_points[:, 1] +
                      c * outlier_points[:, 2] + d) < 0   # Equation of a plane (Written by Paul Bourke) https://paulbourke.net/geometry/pointlineplane/
        below_indices = outlier_indices[below_mask]

        filtered_pos = pc.pos[below_indices]
        new_pc = {'pos': filtered_pos}

        # color handling
        if hasattr(pc, 'x') and pc.x is not None:
            new_pc['x'] = pc.x[below_indices]

        return Data(**new_pc)