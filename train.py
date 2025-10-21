import logging

import hydra
import numpy as np
import rootutils
import torch
from lightning import Callback, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from environments.datamodule import TrajectoryDataModule
from loggers.wandb import update_wandb_config
from utils.conf import delete_keys_recursively, log_slurm_job_id, setup_resolvers
from utils.instantiators import (
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_loggers,
)
from utils.logging import configure_logging, log_exception_and_finish_wandb
from utils.seeding import get_rng
from utils.torch_conf import configure_torch

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
@log_exception_and_finish_wandb
def train(cfg: DictConfig) -> None:
    # resolve the entire config to catch any errors early
    OmegaConf.resolve(cfg)
    log_slurm_job_id(cfg)

    configure_logging(cfg.python_logging)

    # init wandb first so we can log any info or errors from instantiating dataset and model
    log.debug("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
    update_wandb_config(cfg)

    # configure torch, e.g. set_float32_matmul_precision
    configure_torch(cfg.get("torch"))

    # seeding
    rng = get_rng(cfg)
    seed = rng.integers(np.iinfo(np.uint32).max)
    log.info(f"Seeding pytorch, numpy, random, and workers with seed {seed}")
    seed_everything(seed, workers=True)

    # instantiate dataset
    datamodule: TrajectoryDataModule = instantiate_datamodule(cfg.data)

    # manually run prepare data and setup so we can use dataset specs for model creation
    log.debug("Preparing and setting up data...")
    datamodule.prepare_data(stage="fit")
    datamodule.setup(stage="fit")

    # instantiate agent
    log.debug("Instantiating agent...")
    # recursively delete these fields in config dictionary
    # we want these to be saved to WandB but we don't want them for instantiation
    delete_keys_recursively(cfg.agent, ["name"])
    agent: LightningModule = hydra.utils.instantiate(
        cfg.agent,
        specs=datamodule.specs,
        normalizer=datamodule.normalizer,
        reverse_transform=datamodule.reverse_transform,
    )

    log.debug("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.debug("Instantiating trainer...")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, _target_=Trainer, callbacks=callbacks, logger=logger
    )

    log.info("Starting training...")
    trainer.fit(agent, datamodule=datamodule)

    log.info("Training completed.")

    datamodule.close()


if __name__ == "__main__":
    setup_resolvers()
    train()
