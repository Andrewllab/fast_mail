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


class AlrFurnitureBenchDataset(TrajectoryDataset):
    def __init__(self, *args, **kwargs):
        self._specs = None
        super().__init__(*args, **kwargs)

    def find_raw_files(self) -> list[Path]:
        # Data collector saves files with datetime pattern: YYYY_MM_DD-HH_MM_SS.h5
        files = list(self.root_dir.glob("*.h5"))

        if not files:
            raise FileNotFoundError(
                f"No raw files found in {self.root_dir}. Please check the path."
            )

        files = list(sorted(files, key=keyfunc))
        return files

    def load_from_raw_file(self, filepath: Path) -> TensorDict | list[TensorDict]:
        log.debug(f"Loading trajectories from file {filepath}")

        traj = TensorDict.from_h5(str(filepath))
        
        if self._specs is None:
            log.debug(
                f"Inferring dataset specs by inspecting trajectory from file {filepath}"
            )
            self._load_specs(traj)
        
        # Handle proprioception data
        proprio = traj["obs", "proprioception"]
        robot_state = torch.cat(
            (
                proprio["joint_pos"],  # shape: (T, 7)
                proprio["gripper_pos"].unsqueeze(-1),  # shape: (T, 1)
            ),
            dim=-1,
        )
        
        ee_pose = torch.cat(
            (
                proprio["eef_pos"],  # shape: (T, 3)
                proprio["eef_quat"],  # shape: (T, 4)
            ),
            dim=-1,
        )

        gripper_cam_transform = traj["obs", "gripper_cam", "frames", "dynamic_extrinsics"]

        action_data = traj["actions"]
        action = torch.cat(
            (
                action_data["eef_pos"],  # shape: (T, 3) 
                action_data["eef_quat"],  # shape: (T, 4)
                action_data["gripper_pos"].unsqueeze(-1),  # shape: (T, 1)
            ),
            dim=-1,
        )

        target_ee_pose = torch.cat(
            (
                action_data["eef_pos"],  # shape: (T, 3)
                action_data["eef_quat"],  # shape: (T, 4)
            ),
            dim=-1,
        )

        traj = TensorDict(
            {
                "obs": {
                    "front_left_cam": {
                        "rgb": traj["obs", "left_cam", "frames", "rgb"],
                        "depth": traj["obs", "left_cam", "frames", "depth"],
                    },
                    "front_right_cam": {
                        "rgb": traj["obs", "right_cam", "frames", "rgb"],
                        "depth": traj["obs", "right_cam", "frames", "depth"],
                    },
                    "gripper_cam": {
                        "rgb": traj["obs", "gripper_cam", "frames", "rgb"],
                        "depth": traj["obs", "gripper_cam", "frames", "depth"],
                    },
                    "robot_state": robot_state,
                    "ee_pose": ee_pose,
                    "target_ee_pose": target_ee_pose,
                    "gripper_cam_transform": gripper_cam_transform
                },
                "action": action
            }
        )

        return traj

    def _load_specs(self, data: TensorDict | None = None) -> None:
        if data is None:
            filepath = self.find_raw_files()[0]
            log.debug(
                f"Inferring dataset specs by inspecting trajectory from file {filepath}"
            )
            data = TensorDict.from_h5(str(filepath))

        # static camera front left
        rgb_shape = data["obs", "left_cam", "frames", "rgb"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[-1] == 3
        depth_shape = data["obs", "left_cam", "frames", "depth"].shape
        assert len(depth_shape) == 3
        assert rgb_shape[:-1] == depth_shape
        height, width, channels = rgb_shape[1:]
        intrinsics = data["obs", "left_cam", "meta", "intrinsics"].reshape(3, 3)
        extrinsics = data.get(("obs", "left_cam", "meta", "extrinsics"), None)
        if extrinsics is not None:
            extrinsics = extrinsics.reshape(4, 4)
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
        rgb_shape = data["obs", "right_cam", "frames", "rgb"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[-1] == 3
        depth_shape = data["obs", "right_cam", "frames", "depth"].shape
        assert len(depth_shape) == 3
        assert rgb_shape[:-1] == depth_shape
        height, width, channels = rgb_shape[1:]
        intrinsics = data["obs", "right_cam", "meta", "intrinsics"].reshape(3, 3)
        extrinsics = data.get(("obs", "right_cam", "meta", "extrinsics"), None)
        if extrinsics is not None:
            extrinsics = extrinsics.reshape(4, 4)
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
        rgb_shape = data["obs", "gripper_cam", "frames", "rgb"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[-1] == 3
        depth_shape = data["obs", "gripper_cam", "frames", "depth"].shape
        assert len(depth_shape) == 3
        assert rgb_shape[:-1] == depth_shape
        height, width, channels = rgb_shape[1:]
        intrinsics = data["obs", "gripper_cam", "meta", "intrinsics"].reshape(3, 3)
        extrinsics = torch.eye(4, dtype=torch.float32)

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

        # robot state, Why is gripper two dim?
        joint_pos = data["obs", "proprioception", "joint_pos"]
        assert joint_pos.ndim == 2
        assert joint_pos.shape[-1] == 7
        gripper_pos = data["obs", "proprioception", "gripper_pos"]
        assert gripper_pos.ndim == 1
        # we concatenate joint_pos and gripper_pos to get a shape of (T, 8)
        robot_state = ObsSpec(elem_shape=(8,), time=self.obs_seq_len)

        # end-effector pose
        ee_pos = data["obs", "proprioception", "eef_pos"]
        assert ee_pos.ndim == 2
        assert ee_pos.shape[-1] == 3
        ee_quat = data["obs", "proprioception", "eef_quat"]
        assert ee_quat.ndim == 2
        assert ee_quat.shape[-1] == 4
        # we concatenate ee_pos and ee_quat to get a shape of (T, 7)
        ee_pose = ObsSpec(elem_shape=(7,), time=self.obs_seq_len)
        target_ee_pose = ObsSpec(elem_shape=(7,), time=self.action_seq_len)

        # gripper_cam_transform
        transform = data["obs", "gripper_cam", "frames", "dynamic_extrinsics"]
        assert transform.ndim == 3
        assert transform.shape[-2:] == (4, 4)
        gripper_cam_transform = ObsSpec(elem_shape=(4, 4), time=self.obs_seq_len)

        # actions
        action_eef_pos = data["actions", "eef_pos"]
        assert action_eef_pos.ndim == 2
        assert action_eef_pos.shape[-1] == 3
        action_eef_quat = data["actions", "eef_quat"]
        assert action_eef_quat.ndim == 2
        assert action_eef_quat.shape[-1] == 4
        action_gripper_pose = data["actions", "gripper_pos"]
        assert action_gripper_pose.ndim == 1

        action = ActionSpec(action_dim=8, time=self.action_seq_len)

        self._specs = DataSpecs(
            obs={
                "front_left_cam": left_cam,
                "front_right_cam": right_cam,
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