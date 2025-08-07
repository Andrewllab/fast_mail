import logging
from pathlib import Path

import torch
from tensordict import TensorDict

from environments.base_dataset import TrajectoryDataset, keyfunc
from environments.specs import (
    ActionSpec,
    CameraSpec,
    DataSpecs,
    DepthStream,
    ObsSpec,
    PinholeCameraIntrinsic,
    RGBStream,
)

log = logging.getLogger(__name__)


"""In ROS, the camera is looking down the +Z axis with the +Y axis pointing down,
and +X axis pointing right. This is the convention the point clouds are in after
conversion by `unproject_depth`. On the other hand, the typical world coordinate
system is with +X pointing forward, +Y pointing left, and +Z pointing up. We can
achieve this transformation using the following rotation matrix.

Reference: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.utils.html#isaaclab.utils.math.convert_camera_frame_orientation_convention

(equivalent to T_USD_to_WORLD @ (T_USD_to_ROS)^(-1) in the convention used in Isaac Sim)
"""
ROS_TO_WORLD = [
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0],
]


class AlrFurnitureBenchDataset(TrajectoryDataset):
    def __init__(self, *args, **kwargs):
        self._specs = None
        super().__init__(*args, **kwargs)

    def find_raw_files(self) -> list[Path]:
        # e.g. insert_one_leg_1[_recovery].hdf5
        files = list(self.root_dir.glob("*.hdf5"))

        if not files:
            raise FileNotFoundError(
                f"No raw files found in {self.root_dir}. Please check the path."
            )

        files = list(sorted(files, key=keyfunc))
        return files

    def load_from_raw_file(self, filepath: Path) -> TensorDict | list[TensorDict]:
        log.debug(f"Loading trajectories from file {filepath}")

        traj = TensorDict.from_h5(str(filepath))
        traj = traj["data", "demo_0"]

        if self._specs is None:
            log.debug(
                f"Inferring dataset specs by inspecting trajectory from file {filepath}"
            )
            self._load_specs(traj)

        robot_state = torch.cat(
            (
                traj["obs", "proprioception", "joint_pos"],  # shape: (T, 7)
                traj["obs", "proprioception", "gripper_closure"],  # shape: (T, 2)
            ),
            dim=-1,
        )

        ee_pose = torch.cat(
            (
                traj["obs", "proprioception", "eef_pos_w"],  # shape: (T, 3)
                traj["obs", "proprioception", "eef_quat_w"],  # shape: (T, 4)
            ),
            dim=-1,
        )

        traj = TensorDict(
            {
                "obs": {
                    "left_cam": {
                        "rgb": traj["obs", "front_left_cam", "rgb"],
                        "depth": traj["obs", "front_left_cam", "depth"].squeeze(-1),
                    },
                    "right_cam": {
                        "rgb": traj["obs", "front_right_cam", "rgb"],
                        "depth": traj["obs", "front_right_cam", "depth"].squeeze(-1),
                    },
                    "gripper_cam": {
                        "rgb": traj["obs", "gripper_cam", "rgb"],
                        "depth": traj["obs", "gripper_cam", "depth"].squeeze(-1),
                    },
                    "robot_state": robot_state,
                    "ee_pose": ee_pose,
                    "target_ee_pose": traj["actions", "action"][..., :7],
                    "gripper_cam_transform": traj[
                        "obs", "gripper_cam", "homogenious_matrix"
                    ].view(-1, 4, 4),
                },
                "action": traj["actions", "action"],
            },  # type: ignore
        )

        # set batch_size in TensorDict, otherwise it can't be indexed
        traj.auto_batch_size_(batch_dims=1)

        return traj

    def _load_specs(self, data: TensorDict | None = None) -> None:
        if data is None:
            filepath = self.find_raw_files()[0]
            log.debug(
                f"Inferring dataset specs by inspecting trajectory from file {filepath}"
            )
            data = TensorDict.from_h5(str(filepath))
            data = data["data", "demo_0"]

        # static camera front left
        rgb_shape = data["obs", "front_left_cam", "rgb"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[-1] == 3
        depth_shape = data["obs", "front_left_cam", "depth"].shape
        assert len(depth_shape) == 4
        assert depth_shape[-1] == 1
        assert rgb_shape[:-1] == depth_shape[:-1]
        height, width, channels = rgb_shape[1:]
        intrinsics = data["obs", "front_left_cam", "intrinsic_matrix"].reshape(3, 3)
        extrinsics = data["obs", "front_left_cam", "homogenious_matrix"].reshape(4, 4)
        # # correct extrinsics by adding conversion from ROS to WORLD camera convention
        # # we right-multiply, since we first need to transform the points
        # # into the WORLD convention, and then apply the extrinsics
        # extrinsics[:3, :3] = extrinsics[:3, :3] @ torch.tensor(
        #     ROS_TO_WORLD, dtype=extrinsics.dtype
        # )
        left_cam = CameraSpec(
            streams={
                "rgb": RGBStream(height, width, channels, channel_order="HWC"),
                "depth": DepthStream(height, width, orthogonal=True),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=height, width=width
            ),
            extrinsics=extrinsics,
        )

        # static camera front right
        rgb_shape = data["obs", "front_right_cam", "rgb"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[-1] == 3
        depth_shape = data["obs", "front_right_cam", "depth"].shape
        assert len(depth_shape) == 4
        assert depth_shape[-1] == 1
        assert rgb_shape[:-1] == depth_shape[:-1]
        height, width, channels = rgb_shape[1:]
        intrinsics = data["obs", "front_right_cam", "intrinsic_matrix"].reshape(3, 3)
        extrinsics = data["obs", "front_right_cam", "homogenious_matrix"].reshape(4, 4)
        # # correct extrinsics by adding conversion from ROS to WORLD camera convention
        # # we right-multiply, since we first need to transform the points
        # # into the WORLD convention, and then apply the extrinsics
        # extrinsics[:3, :3] = extrinsics[:3, :3] @ torch.tensor(
        #     ROS_TO_WORLD, dtype=extrinsics.dtype
        # )
        right_cam = CameraSpec(
            streams={
                "rgb": RGBStream(height, width, channels, channel_order="HWC"),
                "depth": DepthStream(height, width, orthogonal=True),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=height, width=width
            ),
            extrinsics=extrinsics,
        )

        # gripper camera
        rgb_shape = data["obs", "gripper_cam", "rgb"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[-1] == 3
        depth_shape = data["obs", "gripper_cam", "depth"].shape
        assert len(depth_shape) == 4
        assert depth_shape[-1] == 1
        assert rgb_shape[:-1] == depth_shape[:-1]
        height, width, channels = rgb_shape[1:]
        intrinsics = data["obs", "gripper_cam", "intrinsic_matrix"].reshape(3, 3)
        # gripper_cam_transform provides complete transform to camera
        extrinsics = torch.eye(4, dtype=torch.float32)
        # # correct extrinsics by adding conversion from ROS to WORLD camera convention
        # # we right-multiply, since we first need to transform the points
        # # into the WORLD convention, and then apply the extrinsics
        # extrinsics[:3, :3] = extrinsics[:3, :3] @ torch.tensor(
        #     ROS_TO_WORLD, dtype=extrinsics.dtype
        # )
        gripper_cam = CameraSpec(
            streams={
                "rgb": RGBStream(height, width, channels, channel_order="HWC"),
                "depth": DepthStream(height, width, orthogonal=True),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=height, width=width
            ),
            dynamic_pose_obs_key="gripper_cam_transform",
            # gripper_cam_transform provides complete transform to camera
            extrinsics=extrinsics,
        )

        # robot state
        joint_pos = data["obs", "proprioception", "joint_pos"]
        assert joint_pos.ndim == 2
        assert joint_pos.shape[-1] == 7
        gripper_pos = data["obs", "proprioception", "gripper_closure"]
        assert gripper_pos.ndim == 2
        assert gripper_pos.shape[-1] == 2
        # we concatenate joint_pos and gripper_pos to get a shape of (T, 9)
        robot_state = ObsSpec(elem_shape=(9,), time=self.obs_seq_len)

        # end-effector pose
        ee_pos = data["obs", "proprioception", "eef_pos_w"]
        assert ee_pos.ndim == 2
        assert ee_pos.shape[-1] == 3
        ee_quat = data["obs", "proprioception", "eef_quat_w"]
        assert ee_quat.ndim == 2
        assert ee_quat.shape[-1] == 4
        # we concatenate ee_pos and ee_quat to get a shape of (T, 7)
        ee_pose = ObsSpec(elem_shape=(7,), time=self.obs_seq_len)
        target_ee_pose = ObsSpec(elem_shape=(7,), time=self.action_seq_len)

        # gripper_cam_transform
        transform = data["obs", "gripper_cam", "homogenious_matrix"]
        assert transform.ndim == 2
        assert transform.shape[-1] == 16  # flattened 4x4 matrix
        gripper_cam_transform = ObsSpec(elem_shape=(4, 4), time=self.obs_seq_len)

        # actions
        assert data["actions", "action"].ndim == 2
        assert data["actions", "action"].shape[-1] == 8
        action = ActionSpec(action_dim=8, time=self.action_seq_len)

        self._specs = DataSpecs(
            obs={
                "left_cam": left_cam,
                "right_cam": right_cam,
                "gripper_cam": gripper_cam,
                "robot_state": robot_state,
                "ee_pose": ee_pose,
                "target_ee_pose": target_ee_pose,
                "gripper_cam_transform": gripper_cam_transform,
            },
            action=action,
        )

    def get_specs(self) -> DataSpecs:
        if self._specs is None:
            self._load_specs()
        assert self._specs is not None
        return self._specs
