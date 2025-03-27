from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Iterable, Type

import lightning as L
from tensordict import TensorDict

from utils.logging import warn_once

if TYPE_CHECKING:
    from lightning.pytorch.core.optimizer import LightningOptimizer
    from torch import Tensor
    from torch.nn import Module
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer

    from agents.utils.scaler import Scaler
    from environments.specs import DataSpecs


log = logging.getLogger(__name__)


class BaseAgent(L.LightningModule):
    def __init__(
        self,
        model: Callable[[DataSpecs], Module],
        obs_encoder: Callable[[DataSpecs], Module],
        optimizer: Callable[[Iterable[Tensor]], Optimizer],
        lr_scheduler: Callable[[Optimizer], LRScheduler] | None,
        scaler: Type[Scaler],
        goal_encoder: Callable[[DataSpecs], Module] | None,
        specs: DataSpecs,
        ema_decay: float = 0.0,
    ):
        super().__init__()

        if goal_encoder is not None:
            if specs.goal is None:
                log.warning(
                    f"Attempting to instantiate a goal encoder, but dataset does not provide any goals!"
                )
            goal_encoder = goal_encoder(specs)
            specs = goal_encoder.specs
        self.goal_encoder = goal_encoder

        self._obs_encoder = obs_encoder(specs)
        specs = self._obs_encoder.specs

        self._model = model(specs)

        # BALAZS: refactor, since scalar is sort of transform
        self.scaler = scaler(specs)

        self._optimizer_func = optimizer
        self._lr_scheduler_func = lr_scheduler
        self.ema_decay = ema_decay

    @property
    def model(self) -> Module:
        if self.ema_decay > 0 and not self.training:
            return self._ema_model
        return self._model

    @property
    def obs_encoder(self) -> Module:
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
                self._model,
                multi_avg_fn=get_ema_multi_avg_fn(self.ema_decay),
                # required for using EMA with BatchNorm
                # https://pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html
                use_buffers=True,
            )
            self._ema_obs_encoder = AveragedModel(
                self._obs_encoder,
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
            self._ema_model.update_parameters(self._model)
            self._ema_obs_encoder.update_parameters(self._obs_encoder)

    def encode_goal(self, batch: TensorDict) -> TensorDict:
        # maybe compute language embeddings
        # BALAZS: refactor as transform
        if self.goal_encoder is not None:
            if "goal" in batch.keys():
                # put goal embedding back into batch in case another transform needs it

                batch["goal_embed"] = self.goal_encoder(batch["goal"])
            else:
                warn_once(
                    log,
                    "A language encoder has been instantiated, but the data does not contain a `goal` field!",
                )

        return batch
