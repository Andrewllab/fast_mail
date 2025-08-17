import logging

import hydra
import numpy as np
import rootutils
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from environments.datamodule import TrajectoryDataModule
from loggers.wandb import resolve_checkpoint
from utils.conf import (
    delete_keys_recursively,
    patch_load_from_checkpoint,
    setup_resolvers,
)
from utils.instantiators import (
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_loggers,
)
from utils.logging import configure_logging
from utils.torch_conf import configure_torch

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig) -> None:
    # resolve the entire config to catch any errors early
    OmegaConf.resolve(cfg)

    configure_logging(cfg.python_logging)

    # init wandb first so we can log any info or errors from instantiating dataset and model
    log.debug("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg)

    # configure torch, e.g. set_float32_matmul_precision
    configure_torch(cfg.get("torch"))

    # instantiate dataset
    datamodule: TrajectoryDataModule = instantiate_datamodule(cfg.data)

    # manually run prepare data and setup so we can use dataset specs for model creation
    log.debug("Instantiating datamodule...")
    datamodule.prepare_data()
    datamodule.setup(stage="predict")

    # instantiate agent
    log.debug("Instantiating agent...")
    if checkpoint := resolve_checkpoint(cfg, use_artifact=True):
        # load agent config and specs from checkpoint
        train_cfg, checkpoint_path = checkpoint
        agent_cfg = train_cfg.agent
        # modify the _target_ to point to the module's load_from_checkpoint method
        agent_cfg = patch_load_from_checkpoint(agent_cfg, checkpoint_path)
        datamodule_hparams = {}  # hparams are loaded from checkpoint
    else:
        agent_cfg = cfg.agent
        # without a checkpoint, get the hparams from the datamodule like in train.py
        datamodule_hparams = dict(
            specs=datamodule.specs,
            normalizer=datamodule.normalizer,
            reverse_transform=datamodule.reverse_transform,
        )
    # recursively delete these fields in config dictionary
    # we want these to be saved to WandB but we don't want them for instantiation
    delete_keys_recursively(agent_cfg, ["name"])
    agent: LightningModule = hydra.utils.instantiate(agent_cfg, **datamodule_hparams)

    log.debug("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.debug("Instantiating trainer...")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, _target_=Trainer, callbacks=callbacks, logger=logger
    )

    log.info("Starting prediction loop...")
    trainer.predict(agent, datamodule=datamodule)

    log.info("Prediction loop completed.")


if __name__ == "__main__":
    setup_resolvers()
    main()
