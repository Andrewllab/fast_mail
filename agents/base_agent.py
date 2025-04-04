from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Iterable, Type

import lightning as L

from transforms.base_transform import init_transforms

if TYPE_CHECKING:
    from lightning.pytorch.core.optimizer import LightningOptimizer
    from torch import Tensor
    from torch.nn import Module
    from torch.optim.lr_scheduler import LRScheduler
    from torch.optim.optimizer import Optimizer

    from agents.utils.scaler import Scaler
    from environments.specs import DataSpecs
    from transforms.base_transform import TransformPartial, TransformPartialsDict


log = logging.getLogger(__name__)


class BaseAgent(L.LightningModule):
    def __init__(
        self,
        model: Callable[[DataSpecs], Module],
        obs_encoder: TransformPartialsDict,
        optimizer: Callable[[Iterable[Tensor]], Optimizer],
        lr_scheduler: Callable[[Optimizer], LRScheduler] | None,
        scaler: Type[Scaler],
        goal_encoder: TransformPartial | None,
        specs: DataSpecs,
        ema_decay: float = 0.0,
    ):
        super().__init__()

        # maybe instantiate goal encoder (e.g. clip)
        self.goal_encoder, specs = init_transforms(goal_encoder, specs)

        # instantiate observation encoder (chain of transforms)
        self._obs_encoder, specs = init_transforms(obs_encoder, specs)

        self._model = model(specs)

        # TODO: scaler should return modified specs like transforms
        self.scaler = scaler(specs)

        self._specs = specs
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

    @property
    def specs(self) -> DataSpecs:
        """Make the specs available to the agent for acton sampling."""
        return self._specs

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
