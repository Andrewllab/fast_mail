import logging
from typing import Any, Sequence

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
        # the agent always writes the predicted action back to the batch before
        # calling reverse transforms, so it's always available to us here
        self.dataset.write_actions(batch["action"])

    # The BasePredictionWriter only implements on_predict_batch_end, whereas we
    # also need to write actions during validation and testing. Also that
    # implementation of on_predict_batch_end is specific to the predict loop.
    # Therefore we override it with a version that works in any loop.
    @override
    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self.interval.on_batch:
            return
        # we don't use the batch indices anyway, and they are different between
        # predict/validate/test, so just set them to empty
        # batch_indices = trainer.predict_loop.current_batch_indices
        batch_indices = []
        self.write_on_batch_end(
            trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx
        )

    on_validation_batch_end = on_predict_batch_end
    on_test_batch_end = on_predict_batch_end
