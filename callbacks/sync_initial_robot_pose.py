import logging
from typing import Literal, Sequence

import torch
from lightning.pytorch.callbacks import Callback

from agents.null_agent import NullAgent
from environments.datamodule import TrajectoryDataModule
from environments.gym_env_dataset import GymEnvDataset

log = logging.getLogger(__name__)


class SyncInitialRobotPose(Callback):

    def __init__(
        self,
        space: Literal["joint", "task", "cartesian"],
        cartesian_offset: Sequence[float] | None = None,
    ) -> None:
        self.space = space
        cartesian_offset = cartesian_offset or [0.0, 0.0, 0.0]
        self.cartesian_offset = torch.as_tensor(cartesian_offset)

    def on_predict_epoch_start(self, trainer, pl_module):
        assert isinstance(pl_module, NullAgent)
        replay_data = pl_module.datamodule
        assert isinstance(replay_data, TrajectoryDataModule)

        assert isinstance(trainer.datamodule, TrajectoryDataModule)
        env_dataset = trainer.datamodule.env
        assert isinstance(env_dataset, GymEnvDataset)
        robot_env = env_dataset.env

        first_batch = next(iter(replay_data.predict_dataloader()))

        if self.space == "joint":
            first_joint_pos = first_batch["obs", "robot_state"][..., :7].flatten()

        elif self.space == "task":
            first_ee_pose = first_batch["obs", "ee_pose"]
            first_ee_pos = first_ee_pose[..., :3].flatten()
            wxyz = first_ee_pose[..., 3:].flatten()
            xyzw = torch.cat((wxyz[1:], wxyz[:1]), dim=0)

            first_ee_pos += self.cartesian_offset

            robot_arm = robot_env.unwrapped.get_attr("arm")[0]
            home_joint_pos = robot_arm.home_pose
            first_joint_pos, success = robot_arm.solve_inverse_kinematics(
                first_ee_pos, xyzw, home_joint_pos
            )
            assert success, "Failed to solve inverse kinematics for initial pose"

        log.info(f"Setting robot home pose to {first_joint_pos.tolist()}")

        robot_env.reset(options=dict(home_pose=first_joint_pos))
