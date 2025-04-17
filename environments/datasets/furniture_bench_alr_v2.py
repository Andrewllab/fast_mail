import logging
from pathlib import Path

import torch
from tensordict import TensorDict

from environments.base_dataset import TrajectoryDataset
from environments.specs import (
    ActionSpec,
    CameraSpec,
    DataSpecs,
    DepthStream,
    ObsSpec,
    PinholeCameraIntrinsic,
    RGBStream,
)
from utils.math import euler_xyz_to_quaternion

log = logging.getLogger(__name__)


class AlrFurnitureBenchDataset(TrajectoryDataset):
    def __init__(self, *args, **kwargs):
        self._specs = None
        super().__init__(*args, **kwargs)

    def find_raw_files(self) -> list[Path]:
        # e.g. insert_one_leg_franka_instanceable_1.hdf5
        files = list(
            sorted(self.root_dir.glob("*.hdf5"), key=lambda p: p.stem.split("_")[-1])
        )

        if not files:
            raise FileNotFoundError(
                f"No raw files found in {self.root_dir}. Please check the path."
            )
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
                traj["obs", "proprioception", "gripper_pos"],  # shape: (T, 2)
            ),
            dim=-1,
        )

        # rotation actions are stored as delta euler angles, so we convert to
        # quaternion
        euler_xyz = traj["actions"][..., 3:6]
        quat = euler_xyz_to_quaternion(*euler_xyz.T)
        actions = torch.cat(
            (traj["actions"][..., :3], quat, traj["actions"][..., -1:]), dim=-1
        )

        # we also sneakily add the ee pose to the observation, e.g. for
        # computing absolute desired ee poses
        ee_pose = torch.cat(
            (
                traj["obs", "proprioception", "eef_pos"],  # shape: (T, 3)
                traj["obs", "proprioception", "eef_quat"],  # shape: (T, 4)
            ),
            dim=-1,
        )

        traj = TensorDict(
            {
                "obs": {
                    "left_cam": {
                        "rgb": traj["obs", "rgb", "static_camera_front_left"],
                        "depth": traj[
                            "obs", "depth", "static_camera_front_left"
                        ].squeeze(-1),
                    },
                    "right_cam": {
                        "rgb": traj["obs", "rgb", "static_camera_front_right"],
                        "depth": traj[
                            "obs", "depth", "static_camera_front_right"
                        ].squeeze(-1),
                    },
                    "gripper_cam": {
                        "rgb": traj["obs", "rgb", "gripper_camera"],
                        "depth": traj["obs", "depth", "gripper_camera"].squeeze(-1),
                    },
                    "robot_state": robot_state,
                    "gripper_cam_transform": traj[
                        "obs",
                        "camera_params",
                        "extrinsics",
                        "dynamic",
                        "gripper_camera",
                    ].view(-1, 4, 4),
                    "ee_pose": ee_pose,
                },
                "action": actions,
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
        shape = data["obs", "rgb", "static_camera_front_left"].shape
        assert len(shape) == 4
        assert shape[-1] == 3
        intrinsics = data[
            "obs", "camera_params", "intrinsics", "static_camera_front_left"
        ].reshape(3, 3)
        extrinsics = data[
            "obs", "camera_params", "extrinsics", "static", "static_camera_front_left"
        ].reshape(4, 4)
        left_cam = CameraSpec(
            streams={
                "rgb": RGBStream(shape[1], shape[2], shape[3], channel_order="HWC"),
                "depth": DepthStream(shape[-3], shape[-2], orthogonal=True),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=shape[1], width=shape[2]
            ),
            extrinsics=extrinsics,
        )

        # static camera front right
        shape = data["obs", "rgb", "static_camera_front_right"].shape
        assert len(shape) == 4
        assert shape[-1] == 3
        intrinsics = data[
            "obs", "camera_params", "intrinsics", "static_camera_front_right"
        ].reshape(3, 3)
        extrinsics = data[
            "obs", "camera_params", "extrinsics", "static", "static_camera_front_right"
        ].reshape(4, 4)
        right_cam = CameraSpec(
            streams={
                "rgb": RGBStream(shape[1], shape[2], shape[3], channel_order="HWC"),
                "depth": DepthStream(shape[-3], shape[-2], orthogonal=True),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=shape[1], width=shape[2]
            ),
            extrinsics=extrinsics,
        )

        # gripper camera
        shape = data["obs", "rgb", "gripper_camera"].shape
        assert len(shape) == 4
        assert shape[-1] == 3
        intrinsics = data[
            "obs", "camera_params", "intrinsics", "gripper_camera"
        ].reshape(3, 3)
        gripper_cam = CameraSpec(
            streams={
                "rgb": RGBStream(shape[1], shape[2], shape[3], channel_order="HWC"),
                "depth": DepthStream(shape[-3], shape[-2], orthogonal=True),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=shape[1], width=shape[2]
            ),
            dynamic_pose_obs_key="gripper_cam_transform",
            # gripper_cam_transform provides complete transform to camera
            extrinsics=torch.eye(4),
        )

        # robot state
        joint_pos = data["obs", "proprioception", "joint_pos"]
        assert joint_pos.ndim == 2
        assert joint_pos.shape[-1] == 7
        gripper_pos = data["obs", "proprioception", "gripper_pos"]
        assert gripper_pos.ndim == 2
        assert gripper_pos.shape[-1] == 2
        # we concatenate joint_pos and gripper_pos to get a shape of (T, 9)
        robot_state = ObsSpec(elem_shape=(9,), time=self.obs_seq_len)

        # gripper_cam_transform
        transform = data[
            "obs", "camera_params", "extrinsics", "dynamic", "gripper_camera"
        ]
        assert transform.ndim == 2
        assert transform.shape[-1] == 16  # flattened 4x4 matrix
        gripper_cam_transform = ObsSpec(elem_shape=(4, 4), time=self.obs_seq_len)

        # actions
        assert data["actions"].ndim == 2
        assert data["actions"].shape[-1] == 7
        # we convert euler xyz angles to quaternions, resulting in 8-D actions
        # instead of 7-D
        action = ActionSpec(action_dim=8, time=self.action_seq_len)

        self._specs = DataSpecs(
            obs={
                "left_cam": left_cam,
                "right_cam": right_cam,
                "gripper_cam": gripper_cam,
                "robot_state": robot_state,
                "gripper_cam_transform": gripper_cam_transform,
            },
            action=action,
        )

    def get_specs(self) -> DataSpecs:
        if self._specs is None:
            self._load_specs()
        assert self._specs is not None
        return self._specs
