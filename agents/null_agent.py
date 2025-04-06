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


class NullAgent(BaseAgent):
    """The name of the class makes it sound super cool but actually this agent just does nothing."""

    def __init__(
        self,
        specs,
        obs_encoder: TransformPartialsDict,
        replay_data: DictConfig | None = None,
        replay_fps: float = 30.0,
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
            self.clock = pygame.time.Clock()
            self.fps = replay_fps
        else:
            self.datamodule = None

    def configure_optimizers(self):
        raise NotImplementedError(
            "Null agent is not suitable for training. Use a different agent."
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        # just run any transforms on the batch
        batch = self.obs_encoder(batch)

        if self.datamodule is not None:
            self.clock.tick(self.fps)
            try:
                data = next(self.data_it)
                log.debug(f"Fetching action for timestep {batch_idx}...")
            except StopIteration:
                # gracefully terminate the program
                log.debug(
                    f"Replay dataset exhausted after {batch_idx} timesteps. Exiting..."
                )
                raise KeyboardInterrupt
            actions = data["action"]

        else:
            actions = torch.zeros(
                batch.batch_size + self.specs.action.shape, device=batch.device
            )
        return actions

    # reuse predict_step for test_step
    test_step = predict_step
