import json
import logging
from pathlib import Path
from typing import Literal, Mapping, Sequence, TypedDict

import cv2
import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
import torch_levenberg_marquardt as tlm
from tensordict import TensorDict

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None
Rotation = None

from utils.math import (
    make_pose,
    matrix_to_quaternion,
    pose_inv,
    quaternion_to_matrix,
    unmake_pose,
)


class CamAcquisitionType(TypedDict):
    rgb: np.ndarray  # (H, W, 3)
    left: np.ndarray  # (H, W, 3) or (H, W)
    right: np.ndarray  # (H, W, 3) or (H, W)
    depth: np.ndarray  # (H, W)
    corner_image_uv: np.ndarray  # (M, 1, 2)
    corner_ids: np.ndarray  # (M, 1)
    corner_xyz_obj: np.ndarray  # (M, 1, 3)
    base_pose: np.ndarray  # (4, 4) (optional)


AcquisitionType = dict[str, CamAcquisitionType]


class CamMetadataType(TypedDict):
    camera_cls: str
    static: bool
    stream_name: str
    depth_stream_name: str
    height_width: tuple[int, int]
    intrinsics: dict[str, str | int | float | list[float]]


MetadataType = dict[str, CamMetadataType]

log = logging.getLogger(__name__)


def calibrate_intrinsics(
    acquisitions: Sequence[AcquisitionType], metadata: Mapping
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    # TODO: filter out views that don't have 4 detected points
    # TODO: add previous estimate of intrinsics and dist coeffs
    # TODO: add flags, e.g. symmetric fx/fy, etc.
    cam_names = list(metadata.keys())
    num_cams = len(cam_names)
    num_acquisitions = len(acquisitions)

    all_img_points = {
        cam_name: [
            acquisition[cam_name]["corner_image_uv"] for acquisition in acquisitions
        ]
        for cam_name in cam_names
    }
    all_obj_points = {
        cam_name: [
            acquisition[cam_name]["corner_xyz_obj"] for acquisition in acquisitions
        ]
        for cam_name in cam_names
    }
    intrinsics = np.zeros((num_cams, 3, 3))
    intrinsics[..., 2, 2] = 1  # homogeneous transform matrix
    all_dist_coeffs = {}
    T_cam2board = np.zeros((num_cams, num_acquisitions, 4, 4))
    T_cam2board[..., 3, 3] = 1  # homogeneous transform matrix

    for c, cam_name in enumerate(cam_names):
        error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            all_obj_points[cam_name],
            all_img_points[cam_name],
            metadata[cam_name]["height_width"][::-1],  # (width, height)
            None,  # estimate of camera_matrix
            None,  # estimate of dist_coeffs
        )
        n_points = len(np.concatenate(all_obj_points[cam_name]))
        log.debug(
            f"Calibrated intrinsics for {cam_name} using {n_points} points with reprojection error={error:.6f}"
        )

        intrinsics[c] = camera_matrix
        all_dist_coeffs[cam_name] = dist_coeffs

        r_matrices = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
        r_matrices = np.stack(r_matrices)
        t_vecs = np.squeeze(np.stack(tvecs), axis=-1)

        T_cam2board[c, :, :3, :3] = r_matrices
        T_cam2board[c, :, :3, 3] = t_vecs

    return intrinsics, all_dist_coeffs, T_cam2board


def intrinsics_matrix_from_metadata(cam_metadata: CamMetadataType) -> np.ndarray:
    """Extracts the intrinsics matrix from the camera metadata."""
    intrinsics = np.array(
        [
            [cam_metadata["intrinsics"]["fx"], 0.0, cam_metadata["intrinsics"]["cx"]],
            [0.0, cam_metadata["intrinsics"]["fy"], cam_metadata["intrinsics"]["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intrinsics


def solve_procrustes_extended_kabsch(
    other_coords: np.ndarray,
    world_coords: np.ndarray,
) -> np.ndarray:
    """Fits an affine transform using the "extended" Kabsch algorithm, also
    called partial Procrustes Superimposition. The returned transform converts
    points from other_coords to points in world_coords.

    References:
    https://en.wikipedia.org/wiki/Kabsch_algorithm
    https://en.wikipedia.org/wiki/Procrustes_analysis
    """
    # translate both coordinate sets so that their centroids lie at the
    # origins of their respective coordinate systems
    other_coords_mean = np.mean(other_coords, axis=0)
    world_mean = np.mean(world_coords, axis=0)

    other_coords = other_coords - other_coords_mean
    world_coords = world_coords - world_mean

    # use the Kabsch algorithm to find the best fit rotation
    if Rotation is not None:
        rotation, rms = Rotation.align_vectors(world_coords, other_coords)
        rotation = rotation.as_matrix()
        print(f"RMS error of fit: {rms}")
    else:
        # compute the covariance matrix
        H = np.matmul(world_coords.T, other_coords)
        # use singular value decomposition
        U, s, Vh = np.linalg.svd(H)
        # decide if rotation needs to be corrected to preserve right-handedness
        d = np.sign(np.linalg.det(U @ Vh))
        correction = np.identity(3)
        correction[-1, -1] = d
        # compute optimal rotation
        rotation = U @ correction @ Vh

    # the final transform requires a translation to the origin of the other
    # coordinate system, the best fit rotation, and then a translation to the
    # mean in the world coordinate system
    origin_to_world_mean = np.identity(4)
    origin_to_world_mean[:3, :3] = rotation
    origin_to_world_mean[:3, 3] = world_mean

    translate_to_other_origin = np.identity(4)
    translate_to_other_origin[:3, 3] = -other_coords_mean

    return origin_to_world_mean @ translate_to_other_origin


def solve_procrustes_least_squares(
    other_coords: np.ndarray,
    world_coords: np.ndarray,
) -> np.ndarray:
    """Fits an affine transform using least-squares fitting given pairs of
    matching points. The returned transform converts points from other_coords
    to points in world_coords. This algorithm requires minimum 4 pairs of
    points.

    References:
    https://math.stackexchange.com/questions/613530/understanding-an-affine-transformation/613804
    https://en.m.wikipedia.org/wiki/Scale-invariant_feature_transform#Model_verification_by_linear_least_squares
    """
    if other_coords.shape[0] < 4 or world_coords.shape[0] < 4:
        raise ValueError("Least squares fit requires at least 4 points.")

    transform = np.identity(4)

    # construct left side of equation by expressing other_coords in homogenous
    # coordinates
    A = np.append(other_coords, np.ones((other_coords.shape[0], 1)), axis=1)

    # right side of equation is just the matching coordinates in world space
    b = world_coords

    # solve for model parameters (affine transform) by inverting A and
    # multiplying by b
    solution, residuals, rank, s = np.linalg.lstsq(A, b)

    transform[:3] = solution.T
    return transform


class SceneExtrinsics(nn.Module):
    def __init__(
        self,
        n_cams: int,
        n_acquisitions: int,
        T_base2cam: torch.Tensor | None = None,
        T_base2obj: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        # 7 DOF pose for each camera relative to base (inverted)
        if T_base2cam is not None:
            if len(T_base2cam) != n_cams:
                raise ValueError(
                    f"Expected {n_cams} camera extrinsics, got {len(T_base2cam)}"
                )
            T_cam2base = pose_inv(T_base2cam)
            pos, rot = unmake_pose(T_cam2base)
            pose_cam2base = torch.cat((pos, matrix_to_quaternion(rot)), dim=-1)
        else:
            pose_cam2base = torch.zeros(n_cams, 7)
            pose_cam2base[..., 3] = 1  # set quaternion to (1, 0, 0, 0)
        self.pose_cam2base = nn.Parameter(pose_cam2base)

        # 7 DOF pose for the board at each acquisition
        if T_base2obj is not None:
            if len(T_base2obj) != n_acquisitions:
                raise ValueError(
                    f"Expected {n_acquisitions} object poses, got {len(T_base2obj)}"
                )
            pos, rot = unmake_pose(T_base2obj)
            pose_base2obj = torch.cat((pos, matrix_to_quaternion(rot)), dim=-1)
        else:
            pose_base2obj = torch.zeros(n_acquisitions, 7)
            pose_base2obj[..., 3] = 1  # set quaternion to (1, 0, 0, 0)
        self.pose_base2obj = nn.Parameter(pose_base2obj)

    @property
    def T_base2obj(self) -> torch.Tensor:
        pos = self.pose_base2obj[..., :3]
        rot = quaternion_to_matrix(self.pose_base2obj[..., 3:7])
        return make_pose(pos, rot)

    @property
    def T_cam2base(self) -> torch.Tensor:
        trans = self.pose_cam2base[..., :3]
        rot = quaternion_to_matrix(self.pose_cam2base[..., 3:7])
        return make_pose(trans, rot)

    @property
    def T_base2cam(self) -> torch.Tensor:
        return pose_inv(self.T_cam2base)

    def forward(self, batch: TensorDict) -> torch.Tensor:
        xyz_obj = batch["xyz_obj"]
        acquisition_idx = batch["acquisition_idx"]
        cam_idx = batch["cam_idx"]

        T_robotbase2obj = self.T_base2obj[acquisition_idx]  # (N, 4, 4)
        T_cambase2robotbase = batch["T_ee2base"]
        T_cam2cambase = self.T_cam2base[cam_idx]

        # convert to homogeneous coordinates
        xyz_obj = torch.cat((xyz_obj, torch.ones(len(xyz_obj), 1)), dim=-1)
        # PyTorch matmul wants shapes [..., 3, 3] x [..., 3, 1] -> [..., 3, 1]
        # so we add a dimension and take it away after
        xyz_obj = xyz_obj.unsqueeze(dim=-1)

        xyz_cam = T_cam2cambase @ T_cambase2robotbase @ T_robotbase2obj @ xyz_obj

        # unsqueeze and convert back to Cartesian coordinates
        return xyz_cam.squeeze(dim=-1)[:, :3]


class InputsTargetsDataset(torch.utils.data.Dataset):
    """This dataset just wraps two TensorDicts so that they return a tuple
    when indexed, which is required for torch_levenberg_marquardt.
    """

    def __init__(
        self, inputs: TensorDict | torch.Tensor, targets: TensorDict | torch.Tensor
    ) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple:
        return self.inputs[idx], self.targets[idx]

    def __getitems__(self, indices) -> tuple:
        # both TensorDict and Tensor can handle batches of indices
        return self.inputs[indices], self.targets[indices]


def optimize_euclidean_dynamic_cam(
    acquisitions: Sequence[AcquisitionType],
    intrinsics_matrix: np.ndarray,
    metadata: Mapping,
) -> tuple[np.ndarray, np.ndarray]:
    # TODO: initialize scene extrinsics using estimate of T_cam2board, if given
    cam_names = list(metadata.keys())
    num_cams = len(cam_names)
    num_acquisitions = len(acquisitions)

    all_xyz_obj = []
    all_xyz_cam = []
    all_acquisition_idx = []
    all_cam_idx = []
    all_T_ee2base = []
    for acquisition_idx, acquisition in enumerate(acquisitions):
        for cam_idx, cam_name in enumerate(cam_names):
            cam_data = acquisition[cam_name]

            # (N, 1, 3) -> (N, 3)
            all_xyz_obj.append(np.squeeze(cam_data["corner_xyz_obj"], axis=1))

            # TODO: implement 2 methods: interpolating in depth map and using estimated T_cam2board
            all_xyz_cam.append(
                unproject_image_uv(
                    cam_data["depth"],
                    # (N, 1, 3) -> (N, 3)
                    np.squeeze(cam_data["corner_image_uv"], axis=1),
                    intrinsics_matrix[cam_idx],
                ).astype(
                    np.float32  # gets promoted to float64 because the intrinsics are float64
                )
            )

            n_points = len(cam_data["corner_image_uv"])

            # generate a mask that identifies the acquisition each point belongs to
            all_acquisition_idx.append(np.full(n_points, acquisition_idx))

            # generate a mask that identifies the camera each point belongs to
            all_cam_idx.append(np.full(n_points, cam_idx))

            T_ee2base = torch.eye(4, dtype=torch.float32)
            if "base_pose" in cam_data:
                base_pose = torch.from_numpy(cam_data["base_pose"])
                trans = base_pose[:3]
                rot = quaternion_to_matrix(base_pose[3:7])
                # invert the homogeneous transform to get T_ee2base
                T_ee2base = pose_inv(make_pose(trans, rot))

            all_T_ee2base.append(
                T_ee2base.unsqueeze(dim=0).repeat_interleave(n_points, dim=0)
            )

    inputs = TensorDict(
        {
            "xyz_obj": torch.from_numpy(np.concatenate(all_xyz_obj)),
            "acquisition_idx": torch.from_numpy(np.concatenate(all_acquisition_idx)),
            "cam_idx": torch.from_numpy(np.concatenate(all_cam_idx)),
            "T_ee2base": torch.cat(all_T_ee2base),
        }  # type: ignore
    )
    inputs.auto_batch_size_(batch_dims=1)
    xyz_cam = torch.from_numpy(np.concatenate(all_xyz_cam))
    train_dataset = InputsTargetsDataset(inputs, xyz_cam)
    # train_loader = tlm.utils.FastDataLoader(
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        # because TensorDict can already handle batched indices, there is no
        # need for collation, so we pass the identity function as collate_fn
        collate_fn=lambda x: x,
    )

    model = SceneExtrinsics(num_cams, num_acquisitions)
    module = tlm.training.LevenbergMarquardtModule(
        model=model,
        loss_fn=tlm.loss.MSELoss(),
        learning_rate=1.0,
        attempts_per_step=10,
        solve_method="qr",
    )

    tlm.utils.fit(
        module,
        train_loader,
        epochs=100,
    )

    T_base2cam = model.T_base2cam.detach().numpy()
    T_base2obj = model.T_base2obj.detach().numpy()

    return T_base2cam, T_base2obj


class SceneExtrinsicsAndIntrinsics(nn.Module):
    def __init__(
        self,
        n_cams: int,
        n_acquisitions: int,
        intrinsics_matrix: torch.Tensor,
        T_base2cam: torch.Tensor | None = None,
        T_base2obj: torch.Tensor | None = None,
        optimize_intrinsics: bool = False,
    ) -> None:
        super().__init__()

        if len(intrinsics_matrix) != n_cams:
            raise ValueError(
                f"Expected {n_cams} camera intrinsics, got {len(intrinsics_matrix)}"
            )
        if optimize_intrinsics:
            self._intrinsics = nn.Parameter(intrinsics_matrix)
        else:
            self._intrinsics = intrinsics_matrix

        # 7 DOF pose for each camera relative to base (inverted)
        if T_base2cam is not None:
            if len(T_base2cam) != n_cams:
                raise ValueError(
                    f"Expected {n_cams} camera extrinsics, got {len(T_base2cam)}"
                )
            T_cam2base = pose_inv(T_base2cam)
            pos, rot = unmake_pose(T_cam2base)
            pose_cam2base = torch.cat((pos, matrix_to_quaternion(rot)), dim=-1)
        else:
            pose_cam2base = torch.zeros(n_cams, 7)
            pose_cam2base[..., 3] = 1  # set quaternion to (1, 0, 0, 0)
        self.pose_cam2base = nn.Parameter(pose_cam2base)

        # 7 DOF pose for the board at each acquisition
        if T_base2obj is not None:
            if len(T_base2obj) != n_acquisitions:
                raise ValueError(
                    f"Expected {n_acquisitions} object poses, got {len(T_base2obj)}"
                )
            pos, rot = unmake_pose(T_base2obj)
            pose_base2obj = torch.cat((pos, matrix_to_quaternion(rot)), dim=-1)
        else:
            pose_base2obj = torch.zeros(n_acquisitions, 7)
            pose_base2obj[..., 3] = 1  # set quaternion to (1, 0, 0, 0)
        self.pose_base2obj = nn.Parameter(pose_base2obj)

    @property
    def intrinsics(self) -> torch.Tensor:
        return self._intrinsics

    @property
    def T_base2obj(self) -> torch.Tensor:
        pos = self.pose_base2obj[..., :3]
        rot = quaternion_to_matrix(self.pose_base2obj[..., 3:7])
        return make_pose(pos, rot)

    @property
    def T_cam2base(self) -> torch.Tensor:
        trans = self.pose_cam2base[..., :3]
        rot = quaternion_to_matrix(self.pose_cam2base[..., 3:7])
        return make_pose(trans, rot)

    @property
    def T_base2cam(self) -> torch.Tensor:
        return pose_inv(self.T_cam2base)

    def forward(self, batch: TensorDict) -> torch.Tensor:
        xyz_obj = batch["xyz_obj"]
        acquisition_idx = batch["acquisition_idx"]
        cam_idx = batch["cam_idx"]

        T_robotbase2obj = self.T_base2obj[acquisition_idx]  # (N, 4, 4)
        T_cambase2robotbase = batch["T_ee2base"]
        T_cam2cambase = self.T_cam2base[cam_idx]
        T_intrinsics = self.intrinsics[cam_idx]

        # convert to homogeneous coordinates
        xyz_obj = torch.cat((xyz_obj, torch.ones(len(xyz_obj), 1)), dim=-1)
        # PyTorch matmul wants shapes [..., 3, 3] x [..., 3, 1] -> [..., 3, 1]
        # so we add a dimension and take it away after
        xyz_obj = xyz_obj.unsqueeze(dim=-1)

        xyz_cam = T_cam2cambase @ T_cambase2robotbase @ T_robotbase2obj @ xyz_obj

        z_cam = xyz_cam[:, 2:3]
        xy_cam = xyz_cam[:, :3] / z_cam
        xy_cam[:, 2] = 1.0  # set homogenous coordinate to 1.0

        uv_cam = T_intrinsics @ xy_cam

        # unsqueeze and convert back to Cartesian coordinates
        return uv_cam.squeeze(dim=-1)[:, :2]


def optimize_reprojection(
    acquisitions: Sequence[AcquisitionType],
    intrinsics_matrix: np.ndarray,
    metadata: Mapping,
    T_base2cam: np.ndarray | None = None,
    T_base2obj: np.ndarray | None = None,
    optimize_intrinsics: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam_names = list(metadata.keys())
    num_cams = len(cam_names)
    num_acquisitions = len(acquisitions)

    all_xyz_obj = []
    all_image_uv = []
    all_acquisition_idx = []
    all_cam_idx = []
    all_T_ee2base = []
    for acquisition_idx, acquisition in enumerate(acquisitions):
        for cam_idx, cam_name in enumerate(cam_names):
            cam_data = acquisition[cam_name]

            # (N, 1, 3) -> (N, 3)
            all_xyz_obj.append(np.squeeze(cam_data["corner_xyz_obj"], axis=1))

            # (N, 1, 3) -> (N, 3)
            all_image_uv.append(np.squeeze(cam_data["corner_image_uv"], axis=1))

            n_points = len(cam_data["corner_image_uv"])

            # generate a mask that identifies the acquisition each point belongs to
            all_acquisition_idx.append(np.full(n_points, acquisition_idx))

            # generate a mask that identifies the camera each point belongs to
            all_cam_idx.append(np.full(n_points, cam_idx))

            T_ee2base = torch.eye(4, dtype=torch.float32)
            if "base_pose" in cam_data:
                base_pose = torch.from_numpy(cam_data["base_pose"])
                trans = base_pose[:3]
                rot = quaternion_to_matrix(base_pose[3:7])
                # invert the homogeneous transform to get T_ee2base
                T_ee2base = pose_inv(make_pose(trans, rot))

            all_T_ee2base.append(
                T_ee2base.unsqueeze(dim=0).repeat_interleave(n_points, dim=0)
            )

    inputs = TensorDict(
        {
            "xyz_obj": torch.from_numpy(np.concatenate(all_xyz_obj)),
            "acquisition_idx": torch.from_numpy(np.concatenate(all_acquisition_idx)),
            "cam_idx": torch.from_numpy(np.concatenate(all_cam_idx)),
            "T_ee2base": torch.cat(all_T_ee2base),
        }  # type: ignore
    )
    inputs.auto_batch_size_(batch_dims=1)
    xyz_cam = torch.from_numpy(np.concatenate(all_image_uv))
    train_dataset = InputsTargetsDataset(inputs, xyz_cam)
    # TODO: test with tlm.utils.FastDataLoader, including repeating the dataset
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        # because TensorDict can already handle batched indices, there is no
        # need for collation, so we pass the identity function as collate_fn
        collate_fn=lambda x: x,
    )

    model = SceneExtrinsicsAndIntrinsics(
        num_cams,
        num_acquisitions,
        intrinsics_matrix=torch.from_numpy(intrinsics_matrix).to(torch.float32),
        T_base2cam=torch.from_numpy(T_base2cam) if T_base2cam is not None else None,
        T_base2obj=torch.from_numpy(T_base2obj) if T_base2obj is not None else None,
        optimize_intrinsics=optimize_intrinsics,
    )
    module = tlm.training.LevenbergMarquardtModule(
        model=model,
        loss_fn=tlm.loss.MSELoss(),
        learning_rate=1.0,
        attempts_per_step=10,
        solve_method="qr",
    )

    tlm.utils.fit(
        module,
        train_loader,
        epochs=100,
    )

    intrinsics = model.intrinsics.detach().numpy()
    T_base2cam = model.T_base2cam.detach().numpy()
    T_base2obj = model.T_base2obj.detach().numpy()

    return intrinsics, T_base2cam, T_base2obj


def unproject_image_uv(
    depth: np.ndarray, image_uv: np.ndarray, intrinsics_matrix: np.ndarray
) -> np.ndarray:
    # Compute the 3D coordinates from the 2D pixel coordinates
    u, v = image_uv[..., 0], image_uv[..., 1]
    z = interpolate_depth_bilinear(depth, u, v)

    fx, fy = intrinsics_matrix[..., 0, 0], intrinsics_matrix[..., 1, 1]
    cx, cy = intrinsics_matrix[..., 0, 2], intrinsics_matrix[..., 1, 2]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def interpolate_depth_bilinear(
    depth: np.ndarray, u: float | np.ndarray, v: float | np.ndarray
) -> float | np.ndarray:

    height, width = depth.shape[-2:]
    if (
        np.any(u < 0)
        or np.any(u > width - 1)
        or np.any(v < 0)
        or np.any(v > height - 1)
    ):
        raise IndexError(
            f"u,v coordinates {u},{v} are out of bounds for bilinear interpolation"
        )

    # Get the four neighboring depth values
    u0, v0 = np.floor(u).astype(np.int32), np.floor(v).astype(np.int32)
    d00 = depth[..., v0, u0]  # Top-left
    d10 = depth[..., v0, u0 + 1]  # Top-right
    d01 = depth[..., v0 + 1, u0]  # Bottom-left
    d11 = depth[..., v0 + 1, u0 + 1]  # Bottom-right

    # Compute weights for interpolation
    du, dv = u - u0, v - v0

    # Bilinear interpolation
    val = (
        d00 * (1 - du) * (1 - dv)
        + d10 * du * (1 - dv)
        + d01 * (1 - du) * dv
        + d11 * du * dv
    )

    return val


def convert_extrinsics_convention(
    extrinsics: np.ndarray,
    origin: Literal["opengl", "ros", "world"] = "ros",
    target: Literal["opengl", "ros", "world"] = "world",
) -> np.ndarray:
    if origin == target:
        return extrinsics

    extrinsics = extrinsics.copy()

    if origin != "ros":
        # TODO: implement
        raise NotImplementedError

    if target == "world":
        # In ROS, the camera is looking down the +Z axis with the +Y axis pointing down,
        # and +X axis pointing right. This is the convention the point clouds are in after
        # conversion by `unproject_depth`. On the other hand, the typical world coordinate
        # system is with +X pointing forward, +Y pointing left, and +Z pointing up. We can
        # achieve this transformation using the following rotation matrix.

        # Reference: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#isaaclab.utils.math.convert_camera_frame_orientation_convention

        # (equivalent to T_USD_to_WORLD @ (T_USD_to_ROS)^(-1) in the convention used in Isaac Sim)
        ROS_TO_WORLD = np.ndarray(
            [
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
            ],
            dtype=extrinsics.dtype,
        )
        # correct extrinsics by adding conversion from ROS to WORLD camera convention
        # we right-multiply, since we first need to transform the points
        # into the WORLD convention, and then apply the extrinsics
        extrinsics[..., :3, :3] = extrinsics[..., :3, :3] @ ROS_TO_WORLD
        return extrinsics
    else:
        assert target == "opengl"
        raise NotImplementedError


def render_scene_extrinsics(
    extrinsics: np.ndarray,
    T_base2obj: np.ndarray,
    acquisitions: Sequence[AcquisitionType],
    metadata: MetadataType,
    color_sequence: str = "tab10",
    frame_size: float = 0.1,
    point_size: int = 8,
    width: int = 1920,
    height: int = 1080,
) -> None:
    """Render calibration using open3d.

    Each observed corner is rendered as a point cloud colored by the camera it was observed from.
    The poses of the static cameras are rendered as coordinate frames, while each pose of each
    dynamic camera is rendered as a coordinate frame as well.

    Args:
        static_cameras (dict[str, np.ndarray]): Static camera poses in world frame.
        dynamic_cameras (dict[str, np.ndarray]): Dynamic camera poses in world frame.
        calibration_data (list[dict[str, dict[str, np.ndarray]]]): Calibration data containing
            observed corners for each camera.
        color_sequence (str): Matplotlib color sequence to use for color-coding points.
        frame_size (float): Size of the coordinate frames.
        point_size (int): Size of the points in the point cloud.
        width (int): Width of the rendered window.
        height (int): Height of the rendered window.
    """
    raise NotImplementedError

    import open3d as o3d
    import open3d.core as o3c

    cam_names = list(metadata.keys())

    colors = mpl.color_sequences[color_sequence]
    colors = [np.array(c, dtype=np.float32) for c in colors]  # convert to numpy arrays
    colors_it = iter(colors)

    geometries = {}
    all_xyz_cam = []
    for cam_idx, cam_name in enumerate(cam_names):
        for acquisition_idx, acquisition in enumerate(acquisitions):
            cam_data = acquisition[cam_name]

            # (N, 1, 3) -> (N, 3)
            all_xyz_obj.append(np.squeeze(cam_data["corner_xyz_obj"], axis=1))

            # compute xyz_cam usingestimated T_cam2board
            all_xyz_cam.append(
                unproject_image_uv(
                    cam_data["depth"],
                    # (N, 1, 3) -> (N, 3)
                    np.squeeze(cam_data["corner_image_uv"], axis=1),
                    intrinsics_matrix[cam_idx],
                ).astype(
                    np.float32  # gets promoted to float64 because the intrinsics are float64
                )
            )

    # Create point clouds for each camera
    for camera_name, camera_pose in static_cameras.items():
        color = next(colors_it)

        points_world = []
        for acquisition in calibration_data:
            if camera_name not in acquisition:
                continue

            # add all observed corners to the list of points
            points_world.extend(acquisition[camera_name].values())

        # stack all points together
        points_world = np.stack(points_world, axis=0)

        # transform points to world frame
        points_world = points_world @ camera_pose[:3, :3].T + camera_pose[:3, 3]

        # convert to Open3D point cloud
        pcd = o3d.t.geometry.PointCloud(o3c.Tensor(points_world, dtype=o3c.float32))

        # apply the next color to all points from this camera
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

        # add a coordinate frame for the camera pose
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size, origin=camera_pose[:3, 3]
        )
        frame.rotate(camera_pose[:3, :3], center=camera_pose[:3, 3])
        geometries.append(frame)

        # add a sphere indicating the camera's color
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=frame_size / 4)
        sphere.translate(camera_pose[:3, 3])
        sphere.paint_uniform_color(color)
        geometries.append(sphere)

    for camera_name, static_pose in dynamic_cameras.items():
        color = next(colors_it)

        points_world = []
        for acquisition in calibration_data:
            if camera_name not in acquisition:
                continue

            # for each acquisition, transform the observed corners into world frame
            points = np.stack(list(acquisition[camera_name].values()), axis=0)
            dynamic_pose = acquisition[camera_name + "_pose"]
            camera_pose = static_pose @ dynamic_pose
            points_world.extend(points @ camera_pose[:3, :3].T + camera_pose[:3, 3])

            # add a coordinate frame for the camera pose
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=frame_size / 2, origin=camera_pose[:3, 3]
            )
            frame.rotate(camera_pose[:3, :3], center=camera_pose[:3, 3])
            geometries.append(frame)

            # add a sphere indicating the camera's color
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=frame_size / 8)
            sphere.translate(camera_pose[:3, 3])
            sphere.paint_uniform_color(color)
            geometries.append(sphere)

        # concatenate all points together
        points_world = np.stack(points_world, axis=0)

        # convert to Open3D point cloud
        pcd = o3d.t.geometry.PointCloud(o3c.Tensor(points_world, dtype=o3c.float32))

        # apply the next color to all points from this camera
        pcd.paint_uniform_color(color)
        geometries.append(pcd)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=2 * frame_size, origin=[0, 0, 0]
    )
    geometries.append(origin)

    return geometries

    vis.draw(
        geometries,
        point_size=point_size,
        show_ui=True,
        raw_mode=True,
        width=width,
        height=height,
    )


def flatten_nested_dict(nested_dict: dict) -> dict:
    """Flatten a nested dictionary into a single level dictionary."""
    flat_dict = {}

    def _flatten(d, parent_key=""):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                _flatten(v, new_key)
            else:
                flat_dict[new_key] = v

    _flatten(nested_dict)
    return flat_dict


def unflatten_dict(flat_dict: dict) -> dict:
    """Unflatten a dictionary that was flattened with `flatten_nested_dict`."""
    unflat_dict = {}

    for key, value in flat_dict.items():
        parts = key.split(".")
        d = unflat_dict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return unflat_dict


def save_acquisition(
    acquisition: AcquisitionType, folder: Path, acquisition_id: int
) -> None:
    """Save a single acquisition to a file. Images are saved as PNG files."""
    filepath = folder / f"acquisition_{acquisition_id:03d}.npz"
    flat_acquisition = flatten_nested_dict(acquisition)
    np.savez_compressed(filepath, **flat_acquisition)
    log.info(f"Saved acquisition #{acquisition_id} to {filepath}")

    images_folder = folder / f"images_{acquisition_id:03d}"
    images_folder.mkdir(parents=True, exist_ok=True)
    for cam_name, cam_acquisition in acquisition.items():
        for key, image in cam_acquisition.items():
            if key.startswith("corner_") or key == "base_pose":
                continue
            if not isinstance(image, np.ndarray):
                continue

            img_filepath = images_folder / f"{cam_name}_{key}.png"

            # Convert float images to uint8 for PNG saving
            if np.issubdtype(image.dtype, np.floating):
                image = cv2.normalize(
                    image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
            elif image.dtype == np.uint16:
                # OpenCV can save uint16 PNGs directly
                pass
            elif image.dtype != np.uint8:
                image = image.astype(np.uint8)
            cv2.imwrite(str(img_filepath), image)
            log.debug(f"Saved image {key} to {img_filepath}")


def load_acquisitions(folder: Path) -> list[AcquisitionType]:
    """Load all calibration data from a folder in the order they were saved."""
    data = []
    files = sorted(folder.glob("acquisition_*.npz"), key=lambda f: f.name)
    for file in files:
        with np.load(file) as archive:
            acquisition = {key: archive[key] for key in archive.files}
            acquisition = unflatten_dict(acquisition)
            data.append(acquisition)
    log.info(f"Loaded {len(data)} calibration acquisitions from {folder}")
    return data


def save_metadata(metadata: dict[str, float | np.ndarray], filepath: Path) -> None:
    """Save camera and scene metadata to a file."""
    # serializer = lambda obj: obj.tolist() if isinstance(obj, np.ndarray) else obj

    with filepath.open("w") as f:
        json.dump(metadata, f, indent=4)  # , default=serializer)


def load_metadata(filepath: Path) -> dict[str, float | np.ndarray]:
    """Load camera and scene metadata from a file."""
    with filepath.open("r") as f:
        metadata = json.load(f)
    return metadata


def compare_metadata(
    metadata1: dict[str, int | float | str | list[float]],
    metadata2: dict[str, int | float | str | list[float]],
) -> bool:
    """Compare two metadata dictionaries for equality."""
    result = _dict_equal(metadata1, metadata2)

    if isinstance(result, str):
        keys = result.split(".")
        val1 = metadata1
        val2 = metadata2
        for key in keys:
            val1 = val1[key]
            val2 = val2[key]

        log.warning(
            f"Loaded metadata does not match current metadata at {result}:\n{val1} != {val2}"
        )
        return False

    return True


def _dict_equal(
    dict1: dict[str, int | float | str | list[float]],
    dict2: dict[str, int | float | str | list[float]],
    prefix: str = "",
) -> bool | str:
    """Check if two dictionaries are equal."""
    if dict1.keys() != dict2.keys():
        return prefix[:-1]  # remove trailing dot

    for key, value in dict1.items():
        if isinstance(value, dict):
            if not isinstance(dict2[key], dict):
                return prefix + key

            result = _dict_equal(value, dict2[key], prefix=prefix + key + ".")
            if isinstance(result, str):
                return result

        elif not value == dict2[key]:
            return prefix + key

    return True
