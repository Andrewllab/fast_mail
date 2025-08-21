import logging
from typing import Any, Mapping, Sequence

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import BasePredictionWriter
from typing_extensions import override

from environments.gym_env_dataset import GymEnvDataset

log = logging.getLogger(__name__)


class ActionWriter(BasePredictionWriter):
    def __init__(self, dataset: GymEnvDataset):
        super().__init__(write_interval="batch")
        self.dataset = dataset
        self.stage = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.stage = stage

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
        self.dataset.write_actions(prediction)

    # add a modified version of on_predict_batch_end to handle validation and
    # testing, where the action is unpacked from the outputs dict
    # TODO: why is this needed again?
    @override
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.interval.on_batch:
            return
        batch_indices = trainer.predict_loop.current_batch_indices

        assert isinstance(outputs, dict)
        action = outputs["action"]

        self.write_on_batch_end(
            trainer, pl_module, action, batch_indices, batch, batch_idx, dataloader_idx
        )

    on_test_batch_end = on_validation_batch_end
