import logging
from typing import Literal, Sequence

import torch
from lightning.pytorch.callbacks import Callback

from agents.null_agent import NullAgent
from environments.datamodule import TrajectoryDataModule
from environments.gym_env_dataset import GymEnvDataset
from utils.math import convert_quat

log = logging.getLogger(__name__)


class SyncRobotResetPose(Callback):

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

            log.info(f"Setting robot home pose={first_joint_pos}")
            robot_env.reset(options={"home_pose": first_joint_pos})

        elif self.space == "task":
            first_ee_pose = first_batch["obs", "ee_pose"]
            first_ee_pos = first_ee_pose[..., :3].flatten()
            first_wxyz = first_ee_pose[..., 3:].flatten()
            first_xyzw = convert_quat(first_wxyz, to="xyzw")

            first_ee_pos = first_ee_pos + self.cartesian_offset

            log.info(
                f"Setting robot home pose to position={first_ee_pos} and (wxyz) orientation={first_wxyz}"
            )
            robot_env.reset(
                options={"home_position": first_ee_pos, "home_orientation": first_xyzw}
            )
