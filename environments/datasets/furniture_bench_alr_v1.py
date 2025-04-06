import logging
from pathlib import Path

import torch
from tensordict import TensorDict

from environments.base_dataset import TrajectoryDataset
from environments.specs import (
    ActionSpec,
    DataSpecs,
    PinholeCameraIntrinsic,
    RGBDCameraSpec,
    Spec,
)

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
                traj["proprioception", "joint_pos"],  # shape: (T, 9)
                traj["proprioception", "gripper_pos"],  # shape: (T, 2)
            ),
            dim=-1,
        )

        # temporary hack to remove the 4th and 5th elements that seem to always
        # be zero
        action = torch.cat((traj["actions"][:, :3], traj["actions"][:, 5:]), dim=-1)

        traj = TensorDict(
            {
                "obs": {
                    "left_cam": {
                        "rgb": traj["rgb", "static_camera_front_left"],
                        "depth": traj["depth", "static_camera_front_left"].squeeze(-1),
                    },
                    "right_cam": {
                        "rgb": traj["rgb", "static_camera_front_right"],
                        "depth": traj["depth", "static_camera_front_right"].squeeze(-1),
                    },
                    "gripper_cam": {
                        "rgb": traj["rgb", "gripper_camera"],
                        "depth": traj["depth", "gripper_camera"].squeeze(-1),
                    },
                    "robot_state": robot_state,
                    "gripper_cam_transform": traj[
                        "dynamic_camera_extrinsics", "gripper_camera"
                    ],
                },
                "action": action,
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
        shape = data["rgb", "static_camera_front_left"].shape
        assert len(shape) == 4
        assert shape[-1] == 3
        intrinsics = data["camera_intrinsics", "static_camera_front_left"].reshape(3, 3)
        extrinsics = data[
            "static_camera_extrinsics", "static_camera_front_left"
        ].reshape(4, 4)
        left_cam = RGBDCameraSpec(
            shape=(self.obs_seq_len, *shape[-3:]),
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, width=shape[2], height=shape[1]
            ),
            extrinsics=extrinsics,
            orthogonal=False,
            channel_order="HWC",
        )

        # static camera front right
        shape = data["rgb", "static_camera_front_right"].shape
        assert len(shape) == 4
        assert shape[-1] == 3
        intrinsics = data["camera_intrinsics", "static_camera_front_right"].reshape(
            3, 3
        )
        extrinsics = data[
            "static_camera_extrinsics", "static_camera_front_right"
        ].reshape(4, 4)
        right_cam = RGBDCameraSpec(
            shape=(self.obs_seq_len, *shape[-3:]),
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, width=shape[2], height=shape[1]
            ),
            extrinsics=extrinsics,
            orthogonal=False,
            channel_order="HWC",
        )

        # gripper camera
        shape = data["rgb", "gripper_camera"].shape
        assert len(shape) == 4
        assert shape[-1] == 3
        intrinsics = data["camera_intrinsics", "gripper_camera"].reshape(3, 3)
        gripper_cam = RGBDCameraSpec(
            shape=(self.obs_seq_len, *shape[-3:]),
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, width=shape[2], height=shape[1]
            ),
            dynamic_pose_obs_key="gripper_cam_transform",
            extrinsics=None,  # gripper_cam_transform provides complete transform
            orthogonal=False,
            channel_order="HWC",
        )

        # robot state
        joint_pos = data["proprioception", "joint_pos"]
        assert joint_pos.ndim == 2
        assert joint_pos.shape[-1] == 9
        gripper_pos = data["proprioception", "gripper_pos"]
        assert gripper_pos.ndim == 2
        assert gripper_pos.shape[-1] == 2
        # we concatenate joint_pos and gripper_pos to get a shape of (T, 11)
        robot_state = Spec(shape=(self.obs_seq_len, 11), type="state")

        # actions
        assert data["actions"].ndim == 2
        assert data["actions"].shape[-1] == 7
        # we remove the 4th and 5th elements, leaving only 5
        action = ActionSpec(shape=(self.action_seq_len, 5), type="action")

        self._specs = DataSpecs(
            obs={
                "left_cam": left_cam,
                "right_cam": right_cam,
                "gripper_cam": gripper_cam,
                "robot_state": robot_state,
            },
            action=action,
        )

    def get_specs(self) -> DataSpecs:
        if self._specs is None:
            self._load_specs()
        assert self._specs is not None
        return self._specs
