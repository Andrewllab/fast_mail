from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import open3d as o3d
import torch
from torch_geometric.data import Batch

from transforms.base_transform import KeyMapping, Transform
from utils.pyg import apply_mask

if TYPE_CHECKING:
    from torch import Tensor
    from torch_geometric.data import Data

    from environments.specs import DataSpecs

log = logging.getLogger(__name__)


class CropTablePointCloud(Transform):
    """
    Crop the point cloud to only include points below a plane.

    This transform can also be used on batches of point clouds.

    Args:
        specs (DataSpecs): The data specifications.
        distance_threshold (float): Max distance a point can be from the plane model, and still be considered an inlier.
        ransac_n (int): Number of initial points to be considered inliers in each iteration.
        num_iterations (int): The number of iterations for RANSAC.
        probability (float): The expected probability of finding the optimal plane.
    """

    def __init__(
        self,
        specs: DataSpecs,
        distance_threshold: float,
        ransac_n: int,
        num_iterations: int,
        probability: float = 0.99999999,
        pcd_keys: str | Sequence[str] = "pcd",
    ):
        self.distance_threshold = distance_threshold
        self.ransac_n = ransac_n
        self.num_iterations = num_iterations
        self.probability = probability

        self._specs = specs
        if isinstance(pcd_keys, str):
            pcd_keys = [pcd_keys]
        else:
            pcd_keys = list(pcd_keys)
        self._pcd_keys = pcd_keys

        self._normal = None
        self._d = None

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(in_keys=[("obs", key)], out_keys=[("obs", key)])
            for key in self._pcd_keys
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, data: Data) -> Data:

        # https://archive.ph/XNLup    Beware, this is a medium link.

        if isinstance(data, Batch):
            batch = data.to_data_list()
        else:
            batch = [data]

        for i, elem in enumerate(batch):
            pos = elem.pos
            assert pos is not None

            # Get the plane parameters
            if self._normal is None or self._d is None:
                self._normal, self._d = get_plane_parameters(
                    pos,
                    self.distance_threshold,
                    self.ransac_n,
                    self.num_iterations,
                    self.probability,
                )

            # a point is above the plane if P * N + d >= 0, where * is the dot product
            mask = torch.matmul(pos, self._normal) + self._d >= 0

            elem = apply_mask(elem, mask)
            batch[i] = elem

        if isinstance(data, Batch):
            data = Batch.from_data_list(batch)
        else:
            data = batch[0]

        return data


def get_plane_parameters(
    pos: Tensor,
    distance_threshold: float,
    ransac_n: int,
    num_iterations: int,
    probability: float = 0.99999999,
) -> tuple[Tensor, float]:
    pos_np = pos.detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos_np)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold,
        ransac_n,
        num_iterations,
        probability,
    )

    a, b, c, d = plane_model
    log.debug(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    normal = torch.tensor([a, b, c], device=pos.device)
    d = torch.tensor(d, device=pos.device)

    # Flip the normal vector to point upwards
    if c < 0:
        normal = -normal
        d = -d

    # normalize the normal vector so that we can work in terms of absolute distance
    magnitude = torch.linalg.norm(normal) + 1e-8
    normal, d = normal / magnitude, d / magnitude

    # move the plane upwards by `distance_threshold`
    # to remove points that are too close to the plane
    d -= distance_threshold
    return normal, d
