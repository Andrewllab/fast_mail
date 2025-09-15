# Group 1: Tasks without handcam
# LiftPegUpright, PokeCube, PullCube, PushCube, PickCube, RollBall

import logging
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from environments.base_dataset import TrajectoryDataset, keyfunc
from environments.specs import (  # TextSpec,
    ActionSpec,
    CameraSpec,
    DataSpecs,
    DepthStream,
    EmbedSpec,
    ObsSpec,
    PinholeCameraIntrinsic,
    RGBStream,
)
from utils.math import (
    convert_camera_frame_orientation_convention,
    make_pose,
    matrix_to_quaternion,
    quaternion_to_matrix,
    unmake_pose,
)

log = logging.getLogger(__name__)


def _convert_extrinsics_convention(
    extrinsics_gl: torch.Tensor, target: str = "world"
) -> torch.Tensor:
    """
    Converts a batch of 4x4 extrinsic from opengl to a target (ros or world).
    """
    pos, rot_mat_gl = unmake_pose(extrinsics_gl)

    quat_gl_wxyz = matrix_to_quaternion(rot_mat_gl)

    quat_target_wxyz = convert_camera_frame_orientation_convention(
        quat_gl_wxyz, origin="opengl", target=target
    )

    rot_mat_target = quaternion_to_matrix(quat_target_wxyz)

    extrinsics_target = make_pose(pos, rot_mat_target)

    return extrinsics_target


class ManiSkillDataset(TrajectoryDataset):
    def __init__(self, *args, **kwargs):
        self._specs = None
        super().__init__(*args, **kwargs)

    def find_raw_files(self) -> list[Path]:
        files = list(self.root_dir.glob("*.h5"))

        if not files:
            raise FileNotFoundError(
                f"No raw files found in {self.root_dir}. Please check the path."
            )

        files = list(sorted(files, key=keyfunc))
        return files

    def load_from_raw_file(self, filepath: Path) -> TensorDict:
        log.debug(f"Loading trajectory from file {filepath}")

        # Load the raw data from the file
        traj = TensorDict.from_h5(str(filepath), mode="r")  # readonly

        if self._specs is None:
            log.debug(
                f"Inferring dataset specs by inspecting trajectory from file {filepath}"
            )
            self._load_specs(traj)

        # gripper_cam_extrinsics_gl = traj["obs", "sensor_param", "hand_camera", "cam2world_gl"][:-1]
        # gripper_cam_extrinsics_ros = _convert_extrinsics_convention(
        #     gripper_cam_extrinsics_gl, target="ros"
        # )

        traj = TensorDict(
            {
                "obs": {
                    "base_camera": {
                        "rgb": traj["obs", "sensor_data", "base_camera", "rgb"][:-1],
                        "depth": traj["obs", "sensor_data", "base_camera", "depth"][
                            :-1
                        ].squeeze(-1)
                        / 1000.0,
                    },
                    # "hand_camera": {
                    #     "rgb": traj["obs", "sensor_data", "hand_camera", "rgb"][:-1],
                    #     "depth": traj["obs", "sensor_data", "hand_camera", "depth"][:-1].squeeze(-1)/1000.0,
                    # },
                    "robot_state": traj["obs", "agent", "qpos"][:-1],
                    "ee_pose": traj["obs", "extra", "tcp_pose"][:-1],
                    # "gripper_cam_transform": gripper_cam_extrinsics_ros,
                },
                "action": traj["actions"],
                "goal": {
                    # "text": str(traj["goal", "text"]),
                    "embed": traj["goal", "preprocessed_embedding"],
                },
            },
        )

        return traj

    def _load_specs(self, data: TensorDict | None = None) -> None:
        if data is None:
            filepath = self.find_raw_files()[0]
            log.debug(
                f"Inferring dataset specs by inspecting trajectory from file {filepath}"
            )
            data = TensorDict.from_h5(str(filepath), mode="r")

        rgb_shape = data["obs", "sensor_data", "base_camera", "rgb"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[-1] == 3
        depth_shape = data["obs", "sensor_data", "base_camera", "depth"].shape
        assert len(depth_shape) == 4
        assert depth_shape[-1] == 1
        height, width, channels = rgb_shape[1:]
        intrinsics = data["obs", "sensor_param", "base_camera", "intrinsic_cv"][0]
        extrinsics = data["obs", "sensor_param", "base_camera", "cam2world_gl"][:1]
        extrinsics_ros = _convert_extrinsics_convention(
            extrinsics, target="ros"
        ).squeeze(dim=0)

        base_cam = CameraSpec(
            streams={
                "rgb": RGBStream(height, width, channels, channel_order="HWC"),
                "depth": DepthStream(height, width),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=height, width=width
            ),
            extrinsics=extrinsics_ros,
        )

        # # hand cam
        # rgb_shape = data["obs", "sensor_data", "hand_camera", "rgb"].shape
        # assert len(rgb_shape) == 4
        # assert rgb_shape[-1] == 3
        # depth_shape = data["obs", "sensor_data", "hand_camera", "depth"].shape
        # assert len(depth_shape) == 4
        # assert depth_shape[-1] == 1
        # assert rgb_shape[:-1] == depth_shape[:-1]
        # height, width, channels = rgb_shape[1:]
        # intrinsics = data["obs", "sensor_param", "hand_camera","intrinsic_cv"][0]
        # # gripper_cam_transform provides complete transform to camera
        # extrinsics = torch.eye(4, dtype=torch.float32)

        # hand_cam = CameraSpec(
        #     streams={
        #         "rgb": RGBStream(height, width, channels, channel_order="HWC"),
        #         "depth": DepthStream(height, width),
        #     },
        #     time=self.obs_seq_len,
        #     intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
        #         intrinsics, height=height, width=width
        #     ),
        #     dynamic_pose_obs_key="gripper_cam_transform", # TODO: check if changing this leads to problems
        #     # gripper_cam_transform provides complete transform to camera
        #     extrinsics=extrinsics,
        # )

        # robot state
        joint_pos = data["obs", "agent", "qpos"]
        assert joint_pos.ndim == 2
        assert joint_pos.shape[-1] == 9
        robot_state = ObsSpec(elem_shape=(joint_pos.shape[-1],), time=self.obs_seq_len)

        # end-effector pose
        ee_pose = data["obs", "extra", "tcp_pose"]
        assert ee_pose.ndim == 2
        assert ee_pose.shape[-1] == 7
        # we concatenate ee_pos and ee_quat to get a shape of (T, 7)
        ee_pose = ObsSpec(elem_shape=(ee_pose.shape[-1],), time=self.obs_seq_len)
        target_ee_pose = ObsSpec(
            elem_shape=(ee_pose.shape[-1],), time=self.action_seq_len
        )

        # # gripper_cam_transform
        # transform = data["obs", "sensor_param", "hand_camera", "cam2world_gl"][0]
        # assert transform.ndim == 2
        # assert transform.shape[-2:] == (4, 4)
        # gripper_cam_transform = ObsSpec(elem_shape=(4, 4), time=self.obs_seq_len)

        # actions
        assert data["actions"].ndim == 2
        assert data["actions"].shape[-1] == 8
        action = ActionSpec(
            action_dim=data["actions"].shape[-1], time=self.action_seq_len
        )

        # assert data["goal", "text"].ndim == 0
        # # assert isinstance(data["goal", "text"], str)
        # # assert data["goal", "text"].shape[0] < 78, "Goal text exceeds clips maximum length of 77 characters."
        # text = TextSpec()

        assert data["goal", "preprocessed_embedding"].ndim == 2
        assert data["goal", "preprocessed_embedding"].shape[1] == 1024
        goal = EmbedSpec(
            embed_dim=data["goal", "preprocessed_embedding"].shape[1], n_tokens=1
        )

        self._specs = DataSpecs(
            obs={
                "base_camera": base_cam,
                # "hand_camera": hand_cam,
                "robot_state": robot_state,
                "ee_pose": ee_pose,
                "target_ee_pose": target_ee_pose,
                # "gripper_cam_transform": gripper_cam_transform,
            },
            action=action,
            goal={
                # "text": text,
                "embed": goal,
            },
        )

    def get_specs(self) -> DataSpecs:
        if self._specs is None:
            self._load_specs()
        assert self._specs is not None
        return self._specs
