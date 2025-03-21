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
        model: Callable[[TrajectoryDataset], nn.Module],
        obs_encoder: Callable[[TrajectoryDataset], nn.Module],
        optimizer: Callable[[Iterable[Tensor]], Optimizer],
        lr_scheduler: Callable[[Optimizer], LRScheduler] | None,
        scaler: Type[Scaler],
        language_encoder: nn.Module | None,
        dataset: TrajectoryDataset,
        ema_decay: float = 0.0,
    ):
        super().__init__()

        self._model = model(dataset)
        self._obs_encoder = obs_encoder(dataset)
        self._optimizer_func = optimizer
        self._lr_scheduler_func = lr_scheduler
        self.scaler = scaler(dataset.all_actions)
        self.language_encoder = language_encoder
        self.ema_decay = ema_decay

    @property
    def model(self) -> nn.Module:
        if self.ema_decay > 0 and not self.training:
            return self._ema_model
        return self._model

    @property
    def obs_encoder(self) -> nn.Module:
        if self.ema_decay > 0 and not self.training:
            return self._ema_obs_encoder
        return self._obs_encoder

    def configure_optimizers(self):
        optimizer = self._optimizer_func(self.parameters())

        if self.ema_decay > 0:
            # https://pytorch.org/docs/stable/optim.html#putting-it-all-together-ema

            from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

            # only these models have learnable parameters
            self._ema_model = AveragedModel(
                self.model,
                multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay),
                # required for using EMA with BatchNorm
                # https://pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html
                use_buffers=True,
            )
            self._ema_obs_encoder = AveragedModel(
                self.obs_encoder,
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
            self._ema_model.update_parameters(self)
            self._ema_obs_encoder.update_parameters(self)

    def encode_language(self, obs_dict: dict[str, Tensor]) -> Tensor | None:
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
            return obs_dict["lang_emb"].to(torch.float32)
        else:
            return None
