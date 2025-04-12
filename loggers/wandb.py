from __future__ import annotations

import logging

import wandb
from lightning.pytorch.loggers.wandb import WandbLogger as LightningWandbLogger
from lightning.pytorch.utilities import rank_zero_only
from typing_extensions import override

log = logging.getLogger(__name__)


class WandbLogger(LightningWandbLogger):
    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        if status != "success":
            # If the run was aborted, we should finish it with an error
            # status to avoid any confusion in the UI.
            log.debug("Run was aborted, finishing wandb run with error status.")
            wandb.finish(exit_code=1)

        super().finalize(status)
