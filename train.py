import logging
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from utils.conf import delete_keys_recursively, setup_resolvers
from utils.instantiators import instantiate_callbacks, instantiate_loggers
from utils.logging import configure_logging
from utils.seeding import get_rng, manual_seed
from utils.torch_conf import configure_torch
from utils.wandb import init_wandb

if TYPE_CHECKING:
    from environments.datamodule import TrajectoryDataModule

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.python_logging)

    # init wandb first so we can log any info or errors from instantiating dataset and model
    run = init_wandb(cfg)

    # resolve all interpolated values, so we can remove them if needed
    # this also catches any errors in the config early
    OmegaConf.resolve(cfg)

    # configure torch, e.g. set_float32_matmul_precision
    configure_torch(cfg.get("torch"))

    # seeding
    rng = get_rng(cfg)
    seed = rng.integers(np.iinfo(np.uint32).max)
    log.info(f"Seeding pytorch, numpy, random, and workers with seed {seed}")
    seed_everything(seed, workers=True)

    # instantiate dataset
    # recursively delete these fields in config dictionary
    # we want these to be saved to WandB but we don't want them for instantiation
    delete_keys_recursively(cfg.data, ["name", "task", "task_suite", "randomness"])
    datamodule: TrajectoryDataModule = hydra.utils.instantiate(cfg.data)

    # manually run prepare data and setup so we can use dataset specs for model creation
    log.debug("Loading training data...")
    datamodule.prepare_data()
    datamodule.setup(stage="fit")

    # instantiate agent
    log.debug("Instantiating agent...")
    delete_keys_recursively(cfg.agent, ["name"])
    agent: LightningModule = hydra.utils.instantiate(cfg.agent, specs=datamodule.specs)

    log.debug("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.debug("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.debug("Instantiating trainer...")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, _target_=Trainer, callbacks=callbacks, logger=logger
    )

    log.info("Starting training")
    trainer.fit(agent, datamodule=datamodule)

    log.info("Training done")
    run.finish()


if __name__ == "__main__":
    setup_resolvers()
    main()
