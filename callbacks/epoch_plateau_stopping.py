import logging
from typing import Any, Callable, Optional

import torch
from torch import Tensor

from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl

log = logging.getLogger(__name__)


class EpochPlateauStopping(EarlyStopping):
    """
    Dynamic stopping callback for real-world experiments.
    Stop when validation loss hasn't improved by min_delta for "best_epoch * multiplier" epochs.
    """

    def __init__(self, multiplier: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.multiplier = multiplier
        self.best_epoch = 0

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics

        if (
            trainer.fast_dev_run
            or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
                logs
            )
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_stop, reason = self._evaluate_stopping_criteria(
            current, trainer.current_epoch
        )

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)

    def _evaluate_stopping_criteria(
        self, current: Tensor, current_epoch: int
    ) -> tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(
            current, self.stopping_threshold
        ):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(
            -current, -self.divergence_threshold
        ):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(
            current - self.min_delta, self.best_score.to(current.device)
        ):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.best_epoch = current_epoch
            self.wait_count = 0
        else:
            if current_epoch >= self.best_epoch * self.multiplier:
                self.wait_count += 1
                if self.wait_count >= self.patience:
                    should_stop = True
                    reason = (
                        f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} + {self.best_epoch * self.multiplier} records."
                        f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                    )

        return should_stop, reason
