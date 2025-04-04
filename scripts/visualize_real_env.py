import logging
from typing import TYPE_CHECKING

import hydra
import numpy as np
import rootutils
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from utils.conf import delete_keys_recursively, setup_resolvers
from utils.instantiators import instantiate_callbacks, instantiate_loggers
from utils.logging import configure_logging
from utils.torch_conf import configure_torch

if TYPE_CHECKING:
    from environments.datamodule import TrajectoryDataModule

log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="../configs", config_name="visualize_real_env"
)
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.python_logging)

    # resolve all interpolated values, so we can remove them if needed
    # this also catches any errors in the config early
    OmegaConf.resolve(cfg)

    # configure torch, e.g. set_float32_matmul_precision
    configure_torch(cfg.get("torch"))

    # instantiate dataset
    # recursively delete these fields in config dictionary
    # we want these to be saved to WandB but we don't want them for instantiation
    delete_keys_recursively(cfg.data, ["name", "task", "task_suite", "randomness"])
    datamodule: TrajectoryDataModule = hydra.utils.instantiate(cfg.data)

    # manually run prepare data and setup so we can use dataset specs for model creation
    log.debug("Instantiating real robot environment...")
    datamodule.prepare_data()
    datamodule.setup(stage="predict")

    # instantiate agent
    log.debug("Instantiating agent...")
    delete_keys_recursively(cfg.agent, ["name"])
    agent: LightningModule = hydra.utils.instantiate(cfg.agent, specs=datamodule.specs)

    log.debug("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    callbacks += datamodule.get_callbacks()

    log.debug("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.debug("Instantiating trainer...")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, _target_=Trainer, callbacks=callbacks, logger=logger
    )

    log.info("Starting prediction loop")
    trainer.predict(agent, datamodule=datamodule)


if __name__ == "__main__":
    setup_resolvers()
    main()
