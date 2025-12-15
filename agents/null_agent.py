from __future__ import annotations

import logging
from typing import Sequence

import hydra
import pygame
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict
from torch import Tensor

from agents.base_agent import BaseAgent
from environments.datamodule import TrajectoryDataModule
from environments.specs import DataSpecs
from transforms.base_transform import Compose, Sequential, TransformPartialsDict
from utils.instantiators import instantiate_datamodule

log = logging.getLogger(__name__)


class NullAgent(BaseAgent):
    """The name of the class makes it sound super cool but actually this agent just does nothing."""

    null_action: torch.Tensor

    def __init__(
        self,
        obs_encoder: TransformPartialsDict,
        specs: DataSpecs,
        goal_encoder: TransformPartialsDict | None = None,
        fps: float | None = 30.0,
        null_action: Sequence[float] | None = None,
        replay_data: DictConfig | None = None,
        normalizer: Sequential | None = None,
        reverse_transform: Compose | None = None,
    ):
        obs_encoder = hydra.utils.instantiate(obs_encoder)
        goal_encoder = hydra.utils.instantiate(goal_encoder)

        super().__init__(
            model=lambda specs: None,
            obs_encoder=obs_encoder,
            optimizer=None,
            lr_scheduler=None,
            specs=specs,
            goal_encoder=goal_encoder,
            normalizer=normalizer,
            reverse_transform=reverse_transform,
        )

        if replay_data is not None:
            with open_dict(replay_data):
                replay_data.eval_mode = "dataset"
                replay_data.batch_size = 1  # GymEnvDataset expects batch size of 1
                replay_data.num_workers = 0  # no multiprocessing
                replay_data.pin_memory = False

            log.debug("Instantiating replay dataset...")
            datamodule: TrajectoryDataModule = instantiate_datamodule(replay_data)
            datamodule.prepare_data()
            datamodule.setup(stage="predict")

            if datamodule.specs.action.action_dim != specs.action.action_dim:
                raise ValueError(
                    "Replay dataset action dims do not match agent action dims."
                )

            self.datamodule = datamodule
            self.replay_normalizer = (
                datamodule.normalizer if datamodule.normalizer else Compose()
            )
            self.replay_reverser = (
                datamodule.reverse_transform
                if datamodule.reverse_transform
                else Compose()
            )
            # note: predict_dataloader does not shuffle the data
            self.data_it = iter(datamodule.predict_dataloader())
        else:
            self.register_buffer("null_action", torch.tensor(null_action))
            self.datamodule = None

        self.clock = pygame.time.Clock()
        self.fps = fps

    def configure_optimizers(self):
        raise NotImplementedError(
            "Null agent is not suitable for training. Use a different agent."
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Tensor:
        batch = self.obs_encoder(batch)  # e.g. maybe render the observation

        if self.datamodule is not None:
            try:
                log.debug(f"Fetching action for timestep {batch_idx}...")
                data = next(self.data_it)
            except StopIteration:
                # gracefully terminate the program
                log.debug(
                    f"Replay dataset exhausted after {batch_idx} timesteps. Exiting..."
                )
                raise KeyboardInterrupt

            # simulate an agent that predicts the action, then reverse the transforms
            # using the reverser
            batch["action"] = data["action"]

            # since the reverse transform includes unnormalizing the action,
            # we have to normalize the action before reversing/unnormalizing
            batch = self.replay_normalizer(batch)
            batch = self.replay_reverser.reverse(batch)

        else:
            # if action is already provided in the batch (e.g. from a dataset),
            # we just use that action
            if "action" not in batch:
                actions = torch.zeros(
                    (batch.shape[0], *self.specs.action.shape), device=batch.device
                )
                actions[...] = self.null_action  # broadcasts over leading dimensions
                batch["action"] = actions

            # since the reverse transform includes unnormalizing the action,
            # we have to normalize the action before reversing/unnormalizing
            # TODO: check if this is correct for visualize_real script
            batch = self.normalizer(batch)
            batch = self.reverser.reverse(batch)

        if self.fps is not None:
            self.clock.tick(self.fps)

        return batch["action"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """This method can be used to validate action transforms, since the null agent
        does not modify the actions in any way.
        """

        prediction = self.predict_step(batch, batch_idx)

        if "ref_action" in batch:
            # only if we are validating on demonstration data
            error = F.mse_loss(prediction, batch["ref_action"])
            self.log("val_action_mse", error, batch_size=batch.shape[0])

        # return the prediction in case we want to write it back to the environment
        return prediction

    # validation and testing are identical
    test_step = validation_step
