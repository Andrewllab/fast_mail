import logging
from typing import Any, Sequence

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from lightning.pytorch.trainer.states import TrainerFn

from environments.gym_env_dataset import GymEnvDataset

log = logging.getLogger(__name__)


class ActionWriter(BasePredictionWriter):
    def __init__(self, dataset: GymEnvDataset):
        super().__init__(write_interval="batch")
        self.dataset = dataset
        self.stage = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.stage = stage
        log.debug(f"ActionWriter setup called with stage: {stage}")

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.stage != TrainerFn.FITTING:
            self.dataset.write_actions(prediction)
