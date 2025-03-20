from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Iterable, Type

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch.optim.optimizer import Optimizer

from utils.logging import warn_once

if TYPE_CHECKING:
    from torch import Tensor
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler

    from agents.utils.scaler import Scaler
    from environments.dataset.base_dataset import TrajectoryDataset


log = logging.getLogger(__name__)


class BaseAgent(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Callable[[Iterable[Tensor]], Optimizer],
        lr_scheduler: Callable[[Optimizer], LRScheduler] | None,
        obs_encoder: Callable[[TrajectoryDataset], nn.Module],
        robot_state_encoder: nn.Module | None,
        scaler: Type[Scaler],
        language_encoder: nn.Module | None,
        dataset: TrajectoryDataset,
        ema_decay: float = 0.0,
    ):
        super().__init__()

        self.model = model
        self._optimizer_func = optimizer
        self._lr_scheduler_func = lr_scheduler
        self.ema_decay = ema_decay
        self.obs_encoder = obs_encoder(dataset)
        self.robot_state_encoder = robot_state_encoder
        self.scaler = scaler(dataset.all_actions)
        self.language_encoder = language_encoder

        self.camera_names = dataset.camera_names

    def configure_optimizers(self):
        optimizer = self._optimizer_func(self.parameters())

        if self.ema_decay > 0:
            # https://pytorch.org/docs/stable/optim.html#putting-it-all-together-ema

            from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

            self.ema_model = AveragedModel(
                self,
                multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay),
                # required for using EMA with BatchNorm
                # https://pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html
                use_buffers=True,
            )

        if self._lr_scheduler_func is not None:
            lr_scheduler = self._lr_scheduler_func(optimizer)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer | LightningOptimizer,
        optimizer_closure: Callable[[], Any] | None = None,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        if self.ema_decay > 0:
            self.ema_model.update_parameters(self)

    def encode_obs(self, obs_dict):
        """
        Compute the required embeddings for the visual ones and the latent goal.
        """
        # maybe compute language embeddings
        if self.language_encoder is not None:
            if "lang" in obs_dict:
                obs_dict["lang_emb"] = self.language_encoder(obs_dict["lang"])
            else:
                warn_once(
                    log,
                    "A language encoder has been instantiated, but the data does not contain a `lang` field!",
                )

        # language embeddings might be float16 type
        if "lang_embed" in obs_dict:
            obs_dict["lang_emb"] = obs_dict["lang_emb"].to(torch.float32)

        # compute observation embeddings
        obs_dict["obs"] = self.obs_encoder(obs_dict)

        # maybe compute robot state embeddings
        if self.robot_state_encoder is not None:
            if "robot_state" in obs_dict:
                state_emb = self.robot_state_encoder(obs_dict["robot_state"])
                obs_dict["obs"] = torch.cat([obs_dict["obs"], state_emb], dim=1)
            else:
                warn_once(
                    log,
                    "A robot state encoder has been instantiated, but the data does not contain a `robot_state` field!",
                )

        return obs_dict
