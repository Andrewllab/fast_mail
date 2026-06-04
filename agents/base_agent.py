from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Mapping

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import re

from environments.specs import DataSpecs
from transforms.base_transform import (
    Compose,
    Sequential,
    TransformPartialsDict,
    init_transforms,
)
from utils.legacy import patch_legacy_state_dict

log = logging.getLogger(__name__)


QK_NORM_RE = re.compile(r"(^|\.)(q|k)_(norm|layernorm|ln)\.weight$")

def _is_qk_norm(name: str) -> bool:
    return QK_NORM_RE.search(name) is not None


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
        # weight decays
        qk_norm_weight_decay: float | None = None,  # <-- new
        exclude_norms_from_weight_decay: bool = True,  # <-- new
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

        # weight decays
        self.qk_norm_weight_decay = qk_norm_weight_decay
        self.exclude_norms_from_weight_decay = exclude_norms_from_weight_decay

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
        """Make the specs available to subclasses for action sampling."""
        return self._specs

    @property
    def checkpoint_metadata(self) -> dict[str, Any]:
        return self._checkpoint_metadata

    @checkpoint_metadata.setter
    def checkpoint_metadata(self, metadata: dict[str, Any]) -> None:
        self._checkpoint_metadata = metadata

    def configure_optimizers(self):
        optimizer = self._optimizer_func(self._build_param_groups())

        if self.ema_decay > 0:
            # https://pytorch.org/docs/stable/optim.html#putting-it-all-together-ema

            # only these models have learnable parameters
            # TODO: instead of assuming what submodules have parameters, iterate
            # over children and check which ones have parameters
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

        if (
            self._lr_scheduler_func is not None
            and getattr(self, "_trainer", None) is not None
            and self.trainer.training
        ):
            lr_scheduler = self._lr_scheduler_func(
                optimizer, total_steps=self.trainer.estimated_stepping_batches
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                },
            }
        else:
            return optimizer

    def _build_param_groups(self):
        """Split parameters into groups with different weight decay.

        qk_norm scales get their own decay to prevent runaway growth.
        Other norms, biases, and embeddings get no decay (standard practice).
        Everything else uses the optimizer's default decay from Hydra config.
        """
        qk_norm_decay = getattr(self, "qk_norm_weight_decay", None)
        no_decay_for_norms = getattr(self, "exclude_norms_from_weight_decay", True)

        if qk_norm_decay is None and not no_decay_for_norms:
            # Nothing to split — preserve original behaviour exactly
            return self.parameters()

        qk_norm_params, no_decay_params, decay_params = [], [], []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if qk_norm_decay is not None and _is_qk_norm(name):
                qk_norm_params.append(param)
            elif no_decay_for_norms and (
                name.endswith(".bias")
                or "norm" in name.lower()
                or "_ln." in name
                or name.endswith("_ln.weight")
                or "embedding" in name.lower()
                or name.endswith(".missing_feature_token")
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups = [{"params": decay_params, "name": "decay"}]
        if no_decay_params:
            groups.append(
                {"params": no_decay_params, "weight_decay": 0.0, "name": "no_decay"}
            )
        if qk_norm_params:
            groups.append(
                {
                    "params": qk_norm_params,
                    "weight_decay": qk_norm_decay,
                    "name": "qk_norm_decay",
                }
            )

        log.info(
            "Optimizer param groups: "
            + ", ".join(
                f"{g['name']}={len(g['params'])}"
                f"(wd={g.get('weight_decay', 'default')})"
                for g in groups
            )
        )
        return groups

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
            if not len(self._obs_encoder) == 0:
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
            # TODO: A better solution would be to directly save weights without
            # the _ema_ prefix in the checkpoint, but this would cause issues
            # when resuming training.

            self.configure_optimizers()  # instantiate ema models

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        patch_legacy_state_dict(state_dict, self._obs_encoder)

        if self.ema_decay > 0:
            # duplicate all state dict entries for ema_model and ema_obs_encoder
            # with entries for model and obs_encoder
            for key in list(state_dict.keys()):
                # match "_ema_model.module" so that we don't match _ema_model.n_averaged
                if key.startswith("_ema_model.module"):
                    new_key = key.replace("_ema_model.module.", "_model.")
                    state_dict[new_key] = state_dict[key]

                elif key.startswith("_ema_obs_encoder.module"):
                    new_key = key.replace("_ema_obs_encoder.module.", "_obs_encoder.")
                    state_dict[new_key] = state_dict[key]

        return super().load_state_dict(state_dict, strict, assign)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # values logged here get averaged over an epoch instead of over a step

        # Log info from completed episodes, if validating on environment
        # rollouts.
        # Do this before predicting the next action, just in case.
        if "done" in batch and torch.any(batch["done"]) and "episode_info" in batch:
            done = batch["done"]
            episode_info = batch["episode_info"]
            completed_episodes = episode_info[done]
            # lightning expects scalar values that have already been reduced
            metrics = completed_episodes.float().mean().to_dict()

            assert self.checkpoint_metadata and "epoch" in self.checkpoint_metadata
            self.log_dict(
                {"ckpt_epoch": self.checkpoint_metadata["epoch"], **metrics},
                # since the number of completed episodes can vary between steps,
                # we need to provide lightning with the size of this "batch" of
                # completed episodes for correct averaging
                batch_size=done.sum().item(),
            )

        # predict the next action
        prediction = self.predict_step(batch, batch_idx)

        # Log MSE between predicted and reference actions, if we validating on
        # demonstration data.
        if "ref_action" in batch:
            # only if we are validating on demonstration data
            error = F.mse_loss(prediction, batch["ref_action"])
            self.log("val_action_mse", error, batch_size=batch.shape[0])

        # Return the prediction. If validating on environment rollouts, this
        # will be written back to the environment.
        return prediction

    # validation and testing are identical
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch, batch_idx, dataloader_idx)
