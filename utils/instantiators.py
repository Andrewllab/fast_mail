import logging
from typing import Any, Callable

import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

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


def instantiate_transforms(transforms_cfg: DictConfig) -> list[Callable]:
    """Instantiates transforms from config.

    :param transforms_cfg: A DictConfig object containing transform configurations.
    :return: The instantiated transform, either as a single Callable or wrapped in
    a Compose transform.
    """
    transforms: list[Callable] = []

    if not transforms_cfg:
        log.warning("No transform configs found! Skipping...")
        return transforms

    if not isinstance(transforms_cfg, DictConfig):
        raise TypeError("Transforms config must be a DictConfig!")

    # sort dictionary of transforms by the first part of the key, which should be a number
    def item_to_sort_key(item: tuple[str, Any]) -> float:
        key, _ = item
        # get the part before the first "_"
        num = key.split("_")[0]
        # convert e.g. 1-1 or 1,1 to 1.1, which can be converted to a float
        # periods are not allowed in keys
        num = num.replace("-", ".").replace(",", ".")
        try:
            return float(num)
        except ValueError:
            raise ValueError(
                f"All transform keys must begin with a number separated by an underscore. Got {key}"
            )

    transforms_cfg = dict(sorted(transforms_cfg.items(), key=item_to_sort_key))

    i = 1
    for key, t_conf in transforms_cfg.items():
        if isinstance(t_conf, DictConfig) and "_target_" in t_conf:
            name = key.split("_", maxsplit=1)[1]
            log.debug(f"Instantiating transform #{i} '{name}': <{t_conf._target_}>")
            i += 1
            transforms.append(hydra.utils.instantiate(t_conf))

    return transforms
