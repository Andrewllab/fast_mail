from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pygame
import torch
from omegaconf import DictConfig, open_dict

from agents.base_agent import BaseAgent
from utils.instantiators import instantiate_datamodule

if TYPE_CHECKING:
    from torch import Tensor

    from environments.datamodule import TrajectoryDataModule
    from transforms.base_transform import TransformPartialsDict

log = logging.getLogger(__name__)


def pre_process_actions(delta_pose: torch.Tensor, gripper_command: bool) -> torch.Tensor:
    """Pre-process actions for the environment."""
    # resolve gripper command
    gripper_vel = torch.zeros(delta_pose.shape[0], 1, device=delta_pose.device)
    gripper_vel[:] = -1.0 if gripper_command else 1.0
    # compute actions
    return torch.concat([delta_pose, gripper_vel], dim=1)


class NullAgent(BaseAgent):
    """The name of the class makes it sound super cool but actually this agent just does nothing."""

    def __init__(
        self,
        specs,
        obs_encoder: TransformPartialsDict,
        fps: float | None = 30.0,
        replay_data: DictConfig | None = None,
        noise_model = None,
    ):
        super().__init__(
            model=lambda specs: None,
            obs_encoder=obs_encoder,
            optimizer=None,
            lr_scheduler=None,
            scaler=lambda specs: None,
            goal_encoder=None,
            specs=specs,
        )

        if replay_data is not None:
            with open_dict(replay_data):
                replay_data.eval_mode = "dataset"
                replay_data.batch_size = 1  # GymEnvDataset expects batch size of 1
                replay_data.num_workers = 0  # no multiprocessing
                replay_data.pin_memory = False

            log.debug("Instantiating replay dataset...")
            self.datamodule: TrajectoryDataModule = instantiate_datamodule(replay_data)
            self.datamodule.prepare_data()
            self.datamodule.setup(stage="predict")

            if self.datamodule.specs.action.shape != self.specs.action.shape:
                raise ValueError(
                    "Replay dataset action shape does not match agent action shape."
                )

            self.data_it = iter(self.datamodule.predict_dataloader())
        else:
            self.datamodule = None

        self.clock = pygame.time.Clock()
        self.fps = fps
        self.sensitivity = 0.5

        from isaaclab.devices import Se3Keyboard
        self.teleop_interface = Se3Keyboard(
            pos_sensitivity = 0.1 * self.sensitivity,
            rot_sensitivity = 0.8 * self.sensitivity,
        )
        # self.teleop_interface.add_callback("R", self.reset_recording_instance)
        print(f"KOMMST DU HIER???")



    def reset_recording_instance(self):
        self.should_reset_recording_instance = True


    def configure_optimizers(self):
        raise NotImplementedError(
            "Null agent is not suitable for training. Use a different agent."
        )

    # # reset teleop if env is reset
    # def teleop_reset(self, env, env_ids):
    #     self.teleop_interface.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        # just run any transforms on the batch
        batch = self.obs_encoder(batch)

        # get keyboard command
        delta_pose, gripper_command = self.teleop_interface.advance()
        delta_pose = delta_pose.astype("float32")
        # convert to torch
        delta_pose = torch.tensor(delta_pose, device="cuda") # .repeat(1, 1)
        # pre-process actions
        actions = pre_process_actions(delta_pose, gripper_command)
        print(f"ACTIONS: {actions}")
        if self.fps is not None:
            self.clock.tick(self.fps)

        return actions

    # reuse predict_step for test_step
    test_step = predict_step
