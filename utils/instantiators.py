from __future__ import annotations

import logging

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

from loggers.wandb import init_wandb_logger

log = logging.getLogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        log.info("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.debug(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(cfg: DictConfig) -> list[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    from lightning.pytorch.loggers.wandb import WandbLogger

    logger_cfg = cfg.get("logger", None) or {}
    logger: list[Logger] = []

    if not logger_cfg:
        log.info("No logger configs found! Skipping...")
        return logger

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            log.debug(f"Instantiating logger <{lg_conf._target_}>")
            logger.append(hydra.utils.instantiate(lg_conf))

    wandb_logger = next((lg for lg in logger if isinstance(lg, WandbLogger)), None)
    if wandb_logger is not None:
        init_wandb_logger(wandb_logger, cfg)

    return logger


def instantiate_datamodule(datamodule_cfg: DictConfig) -> "TrajectoryDataModule":
    """Instantiates a data module from config.

    :param datamodule_cfg: A DictConfig object containing data module configurations.
    :return: An instantiated data module.
    """
    from environments.datamodule import TrajectoryDataModule

    if not isinstance(datamodule_cfg, DictConfig):
        raise TypeError("Data module config must be a DictConfig!")

    # delete these fields in config dictionary
    # we want these to be saved to WandB but we don't want them for instantiation
    with open_dict(datamodule_cfg):
        for key in ["name", "task", "task_suite", "randomness"]:
            datamodule_cfg.pop(key, None)

    log.debug("Instantiating <TrajectoryDataModule>")

    # do not instantiate env_dataset recursively, as it may import simulation
    # modules that are not available in the current environment
    with open_dict(datamodule_cfg):
        env_cfg = datamodule_cfg.pop("env_dataset", None)
        datamodule_cfg = hydra.utils.instantiate(datamodule_cfg, _convert_="all")
        datamodule_cfg["env_dataset"] = env_cfg

    datamodule: TrajectoryDataModule = hydra.utils.instantiate(
        datamodule_cfg, _target_=TrajectoryDataModule, _recursive_=False
    )

    return datamodule
