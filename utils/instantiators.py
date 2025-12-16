from __future__ import annotations

import logging

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

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


def instantiate_loggers(logger_cfg: DictConfig) -> list[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
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

    return logger


def instantiate_datamodule(datamodule_cfg: DictConfig) -> "TrajectoryDataModule":
    """Instantiates a data module from config.

    :param datamodule_cfg: A DictConfig object containing data module configurations.
    :return: An instantiated data module.
    """
    from environments.datamodule import TrajectoryDataModule

    if not isinstance(datamodule_cfg, DictConfig):
        raise TypeError("Data module config must be a DictConfig!")

    log.debug("Instantiating TrajectoryDataModule...")

    # resolve entire config before we start modifying it
    OmegaConf.resolve(datamodule_cfg)

    with open_dict(datamodule_cfg):
        # delete these fields in config dictionary
        # we want these to be saved to WandB but we don't want them for instantiation
        for key in ["name", "task", "task_suite", "randomness"]:
            datamodule_cfg.pop(key, None)

        # do not instantiate dataset config, as we need to manipulate the config first
        dataset_cfg = datamodule_cfg.pop("dataset", None)

        # do not instantiate env config recursively, as it may import simulation
        # modules that are not available in the current environment
        env_cfg = datamodule_cfg.pop("env", None)

        # instantiate the config for the datamodule, which creates dictionaries
        # of partials for the transforms
        datamodule_cfg = hydra.utils.instantiate(datamodule_cfg)

        # now put the env config back, so the datamodule can instantiate it
        datamodule_cfg.dataset = dataset_cfg
        datamodule_cfg.env = env_cfg

    # instantiate the TrajectoryDataModule itself, but not recursively, which
    # again ensures that the env and dataset are not instantiated yet
    datamodule: TrajectoryDataModule = hydra.utils.instantiate(
        datamodule_cfg, _target_=TrajectoryDataModule, _recursive_=False
    )

    return datamodule


def get_dataset_class(dataset_cfg: DictConfig) -> "type[TrajectoryDataset]":
    if "_target_" in dataset_cfg:
        # This is the case when instantiating raw datasets. The dataset_cfg
        # should be a valid instantiable config, and we don't need to remove
        # any keys.
        return hydra.utils.get_object(dataset_cfg._target_)

    if "backend" not in dataset_cfg:
        raise ValueError(
            "Dataset config must have either a '_target_' or 'backend' field!"
        )

    backend = dataset_cfg.pop("backend")

    if backend.lower() == "hdf5":
        from environments.base_dataset import Hdf5Dataset

        return Hdf5Dataset

    if backend.lower() == "memmap":
        from environments.base_dataset import MemmapDataset

        return MemmapDataset

    # This is the case when using a custom backend for preprocessing. We cannot
    # specify the class using _target_ here, otherwise hydra will try to
    # instantiate it (including the preprocess transforms) while instantiating
    # the datamodule, which we want to avoid.
    return hydra.utils.get_object(backend)
