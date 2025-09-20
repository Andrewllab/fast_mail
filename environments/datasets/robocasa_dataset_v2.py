import json
import logging
from pathlib import Path

import h5py
import numpy as np
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
    PointMapStream,
    RGBStream,
)
from utils.math import make_pose, matrix_from_quat

log = logging.getLogger(__name__)


class RoboCasaDataset(TrajectoryDataset):
    def __init__(
        self,
        *args,
        trajs_per_task: int | float | None = None,
        **kwargs,
    ):
        self._specs = None
        self.trajs_per_task = trajs_per_task

        super().__init__(*args, **kwargs)

    def find_raw_files(self) -> list[Path]:
        files = list(self.root_dir.glob("**/*.hdf5"))

        if not files:
            raise FileNotFoundError(
                f"No raw files found in {self.root_dir}. Please check the path."
            )

        files = list(sorted(files, key=keyfunc))
        return files

    def load_from_raw_file(self, filepath: Path) -> TensorDict | list[TensorDict]:
        log.debug(f"Loading trajectories from file {filepath}")

        file = h5py.File(str(filepath), "r")
        all_trajs = file["data"]

        # each trajectory is stored under a key like "demo_1", "demo_2", etc.
        demo_keys = list(sorted(all_trajs.keys(), key=lambda demo_i: int(demo_i[5:])))
        # required to extract task desriptions from each demonstration as RoboCasa's task descriptions are not unique: https://robocasa.ai/docs/tasks_scenes_assets/atomic_tasks.html

        if isinstance(self.trajs_per_task, float):
            # if trajs_per_task is a fraction, take that fraction of the total
            # number of trajectories
            end = int(len(demo_keys) * self.trajs_per_task)
        else:
            end = self.trajs_per_task

        trajs = []
        for key in demo_keys[:end]:
            traj = all_trajs[key]

            joint_pos = traj["obs"]["robot0_joint_pos"][...].astype(
                np.float32
            )  # shape: (T, 7), float32
            # gripper joint positions which corresponds to the degree of open/close of the gripper (same as gripper_closure in Isaac)
            gripper_pos = traj["obs"]["robot0_gripper_qpos"][...].astype(
                np.float32
            )  # shape: (T, 2), float32
            ee_pos = traj["obs"]["robot0_eef_pos"][...].astype(
                np.float32
            )  # shape: (T, 3), float32
            ee_quat = traj["obs"]["robot0_eef_quat"][...].astype(
                np.float32
            )  # shape: (T, 4), float32
            # remove joint dims related to static mobile platform
            action = traj["actions"][:, :7].astype(np.float32)

            robot_state = torch.cat(
                (
                    torch.from_numpy(joint_pos),
                    torch.from_numpy(gripper_pos),
                ),
                dim=-1,
            )

            ee_pose = torch.cat(
                (
                    torch.from_numpy(ee_pos),
                    torch.from_numpy(ee_quat),
                ),
                dim=-1,
            )

            camera_poses = traj["camera_params"]["dynamic"]

            traj = TensorDict(
                {
                    "obs": {
                        "left_cam": {
                            "rgb": traj["obs"]["robot0_agentview_left_image"][
                                ...
                            ],  # shape: (T, H, W, 3), uint8
                            "depth": traj["obs"]["robot0_agentview_left_depth"][
                                ..., 0
                            ],  # shape: (T, H, W), float32
                        },
                        "right_cam": {
                            "rgb": traj["obs"]["robot0_agentview_right_image"][
                                ...
                            ],  # shape: (T, H, W, 3), uint8
                            "depth": traj["obs"]["robot0_agentview_right_depth"][
                                ..., 0
                            ],  # shape: (T, H, W), float32
                        },
                        "gripper_cam": {
                            "rgb": traj["obs"]["robot0_eye_in_hand_image"][
                                ...
                            ],  # shape: (T, H, W, 3), uint8
                            "depth": traj["obs"]["robot0_eye_in_hand_depth"][
                                ..., 0
                            ],  # shape: (T, H, W), float32
                        },
                        "ee_pose": ee_pose,  # shape: (T, 7), float32
                        "robot_state": robot_state,  # shape: (T, 9), float64
                        # !!! IMPORTANT: "static" cameras are attached to the robot platform which sometimes moves caused by the robot-arm movements, so they move as well !!!
                        # shape T x 4 x 4 as homogenious matrix
                        "left_cam_transform": camera_poses["robot0_agentview_left"][
                            "extrinsics"
                        ][...],
                        # !!! IMPORTANT: "static" cameras are attached to the robot platform which sometimes moves caused by the robot-arm movements, so they move as well !!!
                        # shape T x 4 x 4 as homogenious matrix
                        "right_cam_transform": camera_poses["robot0_agentview_right"][
                            "extrinsics"
                        ][...],
                        # shape T x 4 x 4 as homogenious matrix
                        "gripper_cam_transform": camera_poses["robot0_eye_in_hand"][
                            "extrinsics"
                        ][...],
                    },
                    "action": action,
                    "goal": {
                        "text": json.loads(traj.attrs["ep_meta"])[
                            "lang"
                        ]  # language description of the current task
                    },
                },  # type: ignore
            )

            trajs.append(traj)

        return trajs

    def _load_specs(self, data: TensorDict | None = None) -> None:
        if data is None:
            filepath = self.find_raw_files()[0]
            log.debug(
                f"Inferring dataset specs by inspecting trajectory from file {filepath}"
            )
            file = h5py.File(str(filepath), "r")
            traj = file["data"]["demo_1"]

        # static left camera
        rgb_shape = traj["obs"]["robot0_agentview_left_image"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[-1] == 3
        depth_shape = traj["obs"]["robot0_agentview_left_depth"].shape
        assert len(depth_shape) == 4
        assert depth_shape[-1] == 1
        assert rgb_shape[:-1] == depth_shape[:-1]
        height, width, channels = rgb_shape[1:]

        # !!! IMPORTANT: "static" cameras are attached to the robot platform which sometimes moves caused by the robot-arm movements, so they move as well !!!
        left_cam_intrinsics = torch.tensor(
            traj["camera_params"]["dynamic"]["robot0_agentview_left"]["intrinsics"]
        ).squeeze(0)
        left_cam_extrinsics = torch.eye(4, dtype=torch.float32)
        left_cam = CameraSpec(
            streams={
                "rgb": RGBStream(height, width, channels, channel_order="HWC"),
                "depth": DepthStream(height, width, orthogonal=True),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                left_cam_intrinsics, height=height, width=width
            ),
            dynamic_pose_obs_key="left_cam_transform",
            extrinsics=left_cam_extrinsics,
        )
        # left_cam_transform
        transform = torch.zeros(1, 16)  # TODO: fill in correct transform
        assert transform.ndim == 2
        assert transform.shape[-1] == 16  # flattened 4x4 matrix
        left_cam_transform = ObsSpec(elem_shape=(4, 4), time=self.obs_seq_len)

        # static right camera
        rgb_shape = traj["obs"]["robot0_agentview_right_image"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[-1] == 3
        depth_shape = traj["obs"]["robot0_agentview_right_depth"].shape
        assert len(depth_shape) == 4
        assert depth_shape[-1] == 1
        assert rgb_shape[:-1] == depth_shape[:-1]
        height, width, channels = rgb_shape[1:]

        # !!! IMPORTANT: "static" cameras are attached to the robot platform which sometimes moves caused by the robot-arm movements, so they move as well !!!
        right_cam_intrinsics = torch.tensor(
            traj["camera_params"]["dynamic"]["robot0_agentview_right"]["intrinsics"]
        ).squeeze(0)
        right_cam_extrinsics = torch.eye(4, dtype=torch.float32)

        right_cam = CameraSpec(
            streams={
                "rgb": RGBStream(height, width, channels, channel_order="HWC"),
                "depth": DepthStream(height, width, orthogonal=True),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                right_cam_intrinsics, height=height, width=width
            ),
            dynamic_pose_obs_key="right_cam_transform",
            extrinsics=right_cam_extrinsics,
        )
        # right_cam_transform
        transform = torch.zeros(1, 16)  # TODO: fill in correct transform
        assert transform.ndim == 2
        assert transform.shape[-1] == 16  # flattened 4x4 matrix
        right_cam_transform = ObsSpec(elem_shape=(4, 4), time=self.obs_seq_len)

        # gripper camera
        rgb_shape = traj["obs"]["robot0_eye_in_hand_image"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[-1] == 3
        depth_shape = traj["obs"]["robot0_eye_in_hand_depth"].shape
        assert len(depth_shape) == 4
        assert depth_shape[-1] == 1
        assert rgb_shape[:-1] == depth_shape[:-1]
        height, width, channels = rgb_shape[1:]

        gripper_cam_intrinsics = torch.tensor(
            traj["camera_params"]["dynamic"]["robot0_eye_in_hand"]["intrinsics"]
        ).squeeze(0)
        gripper_cam_extrinsics = torch.eye(4, dtype=torch.float32)
        gripper_cam = CameraSpec(
            streams={
                "rgb": RGBStream(height, width, channels, channel_order="HWC"),
                "depth": DepthStream(height, width, orthogonal=True),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                gripper_cam_intrinsics, height=height, width=width
            ),
            dynamic_pose_obs_key="gripper_cam_transform",
            # gripper_cam_transform provides complete transform to camera
            extrinsics=gripper_cam_extrinsics,
        )
        # gripper_cam_transform
        transform = torch.zeros(1, 16)  # TODO: fill in correct transform
        assert transform.ndim == 2
        assert transform.shape[-1] == 16  # flattened 4x4 matrix
        gripper_cam_transform = ObsSpec(elem_shape=(4, 4), time=self.obs_seq_len)

        # robot state
        joint_pos = traj["obs"]["robot0_joint_pos"]
        assert joint_pos.ndim == 2
        assert joint_pos.shape[-1] == 7
        gripper_pos = traj["obs"]["robot0_gripper_qpos"]
        assert gripper_pos.ndim == 2
        assert gripper_pos.shape[-1] == 2
        # we concatenate joint_pos and gripper_pos to get a shape of (T, 9)
        robot_state = ObsSpec(elem_shape=(9,), time=self.obs_seq_len)

        # end-effector pose
        ee_pos = traj["obs"]["robot0_eef_pos"]
        assert ee_pos.ndim == 2
        assert ee_pos.shape[-1] == 3
        ee_quat = traj["obs"]["robot0_eef_quat"]
        assert ee_quat.ndim == 2
        assert ee_quat.shape[-1] == 4
        # we concatenate ee_pos and ee_quat to get a shape of (T, 7)
        ee_pose = ObsSpec(elem_shape=(7,), time=self.obs_seq_len)
        target_ee_pose = ObsSpec(elem_shape=(7,), time=self.action_seq_len)

        # !!! NOTE: JOINT-SPACE control actions !!!
        assert traj["actions"].ndim == 2
        # ignore joint dims related to static mobile platform
        assert traj["actions"][:, :7].shape[-1] == 7
        action = ActionSpec(action_dim=7, time=self.action_seq_len)

        self._specs = DataSpecs(
            obs={
                "left_cam": left_cam,
                "right_cam": right_cam,
                "gripper_cam": gripper_cam,
                "robot_state": robot_state,
                "ee_pose": ee_pose,
                "target_ee_pose": target_ee_pose,
                "left_cam_transform": left_cam_transform,
                "right_cam_transform": right_cam_transform,
                "gripper_cam_transform": gripper_cam_transform,
            },
            action=action,
        )

    def get_specs(self) -> DataSpecs:
        if self._specs is None:
            self._load_specs()
        assert self._specs is not None
        return self._specs
