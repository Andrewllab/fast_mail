import logging
from pathlib import Path

import torch
from tensordict import TensorDict

from environments.datasets.base_dataset import TrajectoryDataset
from environments.specs import ActionSpec, CameraSpec, DataSpecs, Spec

log = logging.getLogger(__name__)


class AlrFurnitureBenchDataset(TrajectoryDataset):
    def find_raw_files(self) -> list[Path]:
        # e.g. insert_one_leg_franka_instanceable_1.hdf5
        return list(
            sorted(self.root_dir.glob("*.hdf5"), key=lambda p: p.stem.split("_")[-1])
        )

    def load_from_raw_file(self, filepath: Path) -> TensorDict | list[TensorDict]:
        log.debug(f"Loading trajectories from file {filepath}")

        traj = TensorDict.from_h5(str(filepath))
        traj = traj["data", "demo_0"]

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
                    "front_left_cam": traj["rgb", "static_camera_front_left"],
                    "gripper_cam": traj["rgb", "gripper_camera"],
                    "robot_state": robot_state,
                },
                "action": action,
            },  # type: ignore
        )

        # set batch_size in TensorDict, otherwise it can't be indexed
        traj.auto_batch_size_()

        return traj

    def get_specs(self) -> DataSpecs:
        return DataSpecs(
            obs={
                "front_left_cam": CameraSpec(
                    shape=(self.obs_seq_len, 256, 256, 3), type="rgb"
                ),
                "gripper_cam": CameraSpec(
                    shape=(self.obs_seq_len, 256, 256, 3), type="rgb"
                ),
                "robot_state": Spec(shape=(self.obs_seq_len, 11), type="state"),
            },
            action=ActionSpec(shape=(self.action_seq_len, 5), type="action"),
        )
