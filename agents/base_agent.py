from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

import lightning as L
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from environments.specs import DataSpecs
from transforms.base_transform import (
    Compose,
    Sequential,
    TransformPartial,
    TransformPartialsDict,
    init_transforms,
)

log = logging.getLogger(__name__)


class BaseAgent(L.LightningModule):
    def __init__(
        self,
        model: Callable[[DataSpecs], Module],
        obs_encoder: TransformPartialsDict,
        optimizer: Callable[[Iterable[Tensor]], Optimizer],
        lr_scheduler: Callable[[Optimizer], LRScheduler] | None,
        specs: DataSpecs,
        ema_decay: float = 0.0,
        goal_encoder: TransformPartialsDict | None = None,
        normalizer: Sequential | None = None,
        reverse_transform: Compose | None = None,
    ):
        super().__init__()

        # save the specs and reversible and normalizing transforms to the checkpoints
        # TODO: do not instantiate noise model beforehand, and then save all inputs
        self.save_hyperparameters(
            "specs",
            "normalizer",
            "reverse_transform",
            # don't send these objects to the (WandB) logger since they are not
            # serializable and they are in every checkpoint anyway
            logger=False,
        )

        # maybe instantiate goal encoder (e.g. clip)
        self.goal_encoder, specs = init_transforms(goal_encoder, specs)

        # instantiate observation encoder (chain of transforms)
        self._obs_encoder, specs = init_transforms(obs_encoder, specs)

        self._model = model(specs)

        self._optimizer_func = optimizer
        self._lr_scheduler_func = lr_scheduler
        self._specs = specs
        self.ema_decay = ema_decay
        self.reverser = (
            reverse_transform if reverse_transform is not None else Compose()
        )
        self.normalizer = normalizer if normalizer is not None else Compose()

        # for logging videos to WandB
        self._checkpoint_metadata = {}

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

    @property
    def checkpoint_metadata(self) -> dict[str, Any]:
        return self._checkpoint_metadata

    @checkpoint_metadata.setter
    def checkpoint_metadata(self, metadata: dict[str, Any]) -> None:
        self._checkpoint_metadata = metadata

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

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.ema_decay > 0:
            # we only want to save the ema_model, which is the one we use for inference
            state_dict = checkpoint["state_dict"]
            for key in list(state_dict.keys()):
                if key.startswith("_model.") or key.startswith("_obs_encoder."):
                    del state_dict[key]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if self.ema_decay > 0:
            # we have instantiated the model, but we only have weights for the
            # ema_model, so we have to instantiate a dummy AveragedModel to wrap
            # the model for loading weights
            # TODO: it might be possible to just reorganize the state dict when
            # saving the checkpoint so that the ema_model is unnecessary

            # Don't reinstantiate models if we already have them (e.g. multiple checkpoints)
            if not hasattr(self, "_ema_model"):
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

            # duplicate all state dict entries for ema_model and ema_obs_encoder
            # with entries for model and obs_encoder
            state_dict = checkpoint["state_dict"]
            for key in list(state_dict.keys()):
                if key.startswith("_ema_model.module"):
                    new_key = key.replace("_ema_model.module.", "_model.")
                    state_dict[new_key] = state_dict[key]

                elif key.startswith("_ema_obs_encoder.module"):
                    new_key = key.replace("_ema_obs_encoder.module.", "_obs_encoder.")
                    state_dict[new_key] = state_dict[key]
