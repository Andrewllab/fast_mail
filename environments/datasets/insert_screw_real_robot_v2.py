import logging
from pathlib import Path
from typing import Mapping

import torch
from tensordict import TensorDict

from environments.base_dataset import TrajectoryDataset, keyfunc
from environments.specs import (
    ActionSpec,
    CameraSpec,
    DataSpecs,
    DepthStream,
    ImageStream,
    ObsSpec,
    PinholeCameraIntrinsic,
    RGBStream,
)
from utils.math import convert_quat

log = logging.getLogger(__name__)


class RealRobotDataset(TrajectoryDataset):
    def __init__(
        self,
        *args,
        extrinsics: Mapping[str, list[list[float]]] | None = None,
        lightweight: bool = False,
        **kwargs,
    ):
        self._specs = None
        self.extrinsics_dict = extrinsics
        self.lightweight = (
            lightweight  # Only load rgb and depth streams to not overload memory
        )
        super().__init__(*args, **kwargs)

    def find_raw_files(self) -> list[Path]:
        # Data collector saves files with datetime pattern: YYYY_MM_DD-HH_MM_SS.h5
        # Also check all subfolders:
        files = list(self.root_dir.glob("**/*.h*5"))  # match .h5 and .hdf5

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
                convert_quat(proprio["eef_quat"], to="wxyz"),  # shape: (T, 4)
            ),
            dim=-1,
        )

        gripper_cam_transform = traj[
            "obs", "gripper_cam", "frames", "dynamic_extrinsics"
        ]

        action_data = traj["action"]
        target_ee_pose = torch.cat(
            (
                action_data["eef_pos"],  # shape: (T, 3)
                convert_quat(action_data["eef_quat"], to="wxyz"),  # shape: (T, 4)
            ),
            dim=-1,
        )

        target_joint_pos = action_data["joint_pos"]
        target_gripper_pos = action_data["gripper_pos"].unsqueeze(-1)

        action = torch.cat(
            (
                target_ee_pose,  # shape: (T, 7)
                target_gripper_pos,  # shape: (T, 1)
            ),
            dim=-1,
        )

        td = TensorDict(
            {
                "obs": {
                    "front_left_cam": {
                        "depth": traj["obs", "left_cam", "frames", "depth"],
                        "left": traj["obs", "left_cam", "frames", "left"],
                        "right": traj["obs", "left_cam", "frames", "right"],
                    },
                    "front_right_cam": {
                        "depth": traj["obs", "right_cam", "frames", "depth"],
                        "left": traj["obs", "right_cam", "frames", "left"],
                        "right": traj["obs", "right_cam", "frames", "right"],
                    },
                    "gripper_cam": {
                        "rgb": traj["obs", "gripper_cam", "frames", "rgb"],
                        "depth": traj["obs", "gripper_cam", "frames", "depth"],
                        "left": traj["obs", "gripper_cam", "frames", "left"],
                        "right": traj["obs", "gripper_cam", "frames", "right"],
                    },
                    "robot_state": robot_state,
                    "ee_pose": ee_pose,
                    "target_ee_pose": target_ee_pose,
                    "target_joint_pos": target_joint_pos,
                    "target_gripper_pos": target_gripper_pos,
                    "gripper_cam_transform": gripper_cam_transform.to(
                        dtype=torch.float32
                    ),
                },
                "action": action,
            }  # type: ignore
        )

        if self.lightweight:
            td["obs", "front_left_cam"].pop("right")
            td["obs", "front_right_cam"].pop("right")
            td["obs", "gripper_cam"].pop("left")
            td["obs", "gripper_cam"].pop("right")

        return td

    def _load_specs(self, data: TensorDict | None = None) -> None:
        if data is None:
            filepath = self.find_raw_files()[0]
            log.debug(
                f"Inferring dataset specs by inspecting trajectory from file {filepath}"
            )
            data = TensorDict.from_h5(str(filepath))

        # static camera front left (Zed mini)
        # depth: (T, 720, 1280)
        depth_shape = data["obs", "left_cam", "frames", "depth"].shape
        T = depth_shape[0]
        assert len(depth_shape) == 3
        # left|right: (T, 720, 1280, 3)
        left_shape = data["obs", "left_cam", "frames", "left"].shape
        right_shape = data["obs", "left_cam", "frames", "right"].shape
        assert left_shape == right_shape
        assert left_shape[:-1] == depth_shape
        height, width, channels = left_shape[1:]
        intrinsics = data["obs", "left_cam", "meta", "intrinsics"].reshape(3, 3)
        baseline = data["obs", "left_cam", "meta", "baseline"].item()
        if self.extrinsics_dict is not None:
            extrinsics = self.extrinsics_dict["front_left_cam"]["extrinsics"]
            extrinsics = torch.as_tensor(extrinsics, dtype=torch.float32).reshape(4, 4)
        elif extrinsics := data.get(("obs", "left_cam", "meta", "extrinsics"), None):
            extrinsics = extrinsics.reshape(4, 4)
        streams = {
            "left": RGBStream(height, width, channels, channel_order="HWC"),
            "depth": DepthStream(height, width, orthogonal=True),
        }
        if not self.lightweight:
            streams["right"] = RGBStream(height, width, channels, channel_order="HWC")
        left_cam = CameraSpec(
            streams=streams,
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=height, width=width
            ),
            baseline=baseline,
            extrinsics=extrinsics,
        )

        # static camera front right (Zed mini)
        # depth: (T, 720, 1280)
        depth_shape = data["obs", "right_cam", "frames", "depth"].shape
        assert len(depth_shape) == 3
        assert depth_shape[0] == T
        # left|right: (T, 720, 1280, 3)
        left_shape = data["obs", "right_cam", "frames", "left"].shape
        right_shape = data["obs", "right_cam", "frames", "right"].shape
        assert left_shape == right_shape
        assert left_shape[:-1] == depth_shape
        height, width, channels = left_shape[1:]
        intrinsics = data["obs", "right_cam", "meta", "intrinsics"].reshape(3, 3)
        baseline = data["obs", "right_cam", "meta", "baseline"].item()
        if self.extrinsics_dict is not None:
            extrinsics = self.extrinsics_dict["front_right_cam"]["extrinsics"]
            extrinsics = torch.as_tensor(extrinsics, dtype=torch.float32).reshape(4, 4)
        elif extrinsics := data.get(("obs", "right_cam", "meta", "extrinsics"), None):
            extrinsics = extrinsics.reshape(4, 4)
        streams = {
            "left": RGBStream(height, width, channels, channel_order="HWC"),
            "depth": DepthStream(height, width, orthogonal=True),
        }
        if not self.lightweight:
            streams["right"] = RGBStream(height, width, channels, channel_order="HWC")
        right_cam = CameraSpec(
            streams=streams,
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=height, width=width
            ),
            baseline=baseline,
            extrinsics=extrinsics,
        )

        # gripper camera (Realsense D405)
        # rgb: (T, 480, 640, 3)
        rgb_shape = data["obs", "gripper_cam", "frames", "rgb"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[0] == T
        assert rgb_shape[-1] == 3
        # depth: (T, 480, 640)
        depth_shape = data["obs", "gripper_cam", "frames", "depth"].shape
        assert len(depth_shape) == 3
        assert rgb_shape[:-1] == depth_shape
        # left|right: (T, 480, 640)
        left_shape = data["obs", "gripper_cam", "frames", "left"].shape
        right_shape = data["obs", "gripper_cam", "frames", "right"].shape
        assert left_shape == right_shape == depth_shape == rgb_shape[:-1]
        height, width, channels = rgb_shape[1:]
        intrinsics = data["obs", "gripper_cam", "meta", "intrinsics"].reshape(3, 3)
        baseline = data["obs", "gripper_cam", "meta", "baseline"].item()
        if self.extrinsics_dict is not None:
            extrinsics = self.extrinsics_dict["gripper_cam"]["extrinsics"]
            extrinsics = torch.as_tensor(extrinsics, dtype=torch.float32).reshape(4, 4)
        elif extrinsics := data.get(("obs", "gripper_cam", "meta", "extrinsics"), None):
            extrinsics = extrinsics.reshape(4, 4)
        streams = {
            "rgb": RGBStream(height, width, channels, channel_order="HWC"),
            "depth": DepthStream(height, width, orthogonal=True),
        }
        if not self.lightweight:
            streams["left"] = ImageStream(height, width, channel_order="HW")
            streams["right"] = ImageStream(height, width, channel_order="HW")
        gripper_cam = CameraSpec(
            streams=streams,
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=height, width=width
            ),
            baseline=baseline,
            extrinsics=extrinsics,
            dynamic_pose_obs_key="gripper_cam_transform",
        )

        joint_pos = data["obs", "proprioception", "joint_pos"]
        assert joint_pos.ndim == 2
        assert joint_pos.shape[0] == T
        assert joint_pos.shape[-1] == 7
        gripper_pos = data["obs", "proprioception", "gripper_pos"]
        assert gripper_pos.ndim == 1
        # we concatenate joint_pos and gripper_pos to get a shape of (T, 8)
        robot_state = ObsSpec(elem_shape=(8,), time=self.obs_seq_len)

        # end-effector pose
        ee_pos = data["obs", "proprioception", "eef_pos"]
        assert ee_pos.ndim == 2
        assert ee_pos.shape[0] == T
        assert ee_pos.shape[-1] == 3
        ee_quat = data["obs", "proprioception", "eef_quat"]
        assert ee_quat.ndim == 2
        assert ee_quat.shape[-1] == 4
        # we concatenate ee_pos and ee_quat to get a shape of (T, 7)
        ee_pose = ObsSpec(elem_shape=(7,), time=self.obs_seq_len)
        target_ee_pose = ObsSpec(elem_shape=(7,), time=self.action_seq_len)

        target_joint_pos = data["action", "joint_pos"]
        assert target_joint_pos.ndim == 2
        assert target_joint_pos.shape[0] == T
        assert target_joint_pos.shape[-1] == 7
        target_joint_pos = ObsSpec(elem_shape=(7,), time=self.obs_seq_len)

        target_gripper_pos = data["action", "gripper_pos"]
        assert target_gripper_pos.ndim == 1
        assert target_gripper_pos.shape[0] == T
        target_gripper_pos = ObsSpec(elem_shape=(1,), time=self.obs_seq_len)

        # gripper_cam_transform
        transform = data["obs", "gripper_cam", "frames", "dynamic_extrinsics"]
        assert transform.ndim == 3
        assert transform.shape[0] == T
        assert transform.shape[-2:] == (4, 4)
        gripper_cam_transform = ObsSpec(elem_shape=(4, 4), time=self.obs_seq_len)

        # actions
        action_eef_pos = data["action", "eef_pos"]
        assert action_eef_pos.ndim == 2
        assert action_eef_pos.shape[0] == T
        assert action_eef_pos.shape[-1] == 3
        action_eef_quat = data["action", "eef_quat"]
        assert action_eef_quat.ndim == 2
        assert action_eef_quat.shape[0] == T
        assert action_eef_quat.shape[-1] == 4
        action_gripper_pose = data["action", "gripper_pos"]
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
                "target_joint_pos": target_joint_pos,
                "target_gripper_pos": target_gripper_pos,
                "gripper_cam_transform": gripper_cam_transform,
            },
            action=action,
        )

    def get_specs(self) -> DataSpecs:
        if self._specs is None:
            self._load_specs()
        assert self._specs is not None
        return self._specs
    
    def _get_specs(self) -> DataSpecs:
        # This allows us to change the calibration data even if we are already using
        # a prepreprocessed dataset
        if self.from_prepreprocessed:
            self._specs = self.prepreprocess_transforms[-1].specs

            # the specs from the raw dataset are expected not to have lengths
            # so we need to clear them to avoid double counting
            self._specs._lengths.clear()
            
            for cam in ["front_left_cam", "front_right_cam", "gripper_cam"]:
                extrinsics = self.extrinsics_dict[cam]["extrinsics"]
                object.__setattr__(
                    self._specs.obs[cam], 
                    'extrinsics', 
                    torch.as_tensor(extrinsics, dtype=torch.float32).reshape(4, 4)
                )
            
            return self._specs
        return self.get_specs()