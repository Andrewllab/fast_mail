from __future__ import annotations

import logging

import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from wandb.wandb_run import Run

WANDB_LOGGER_KEYS = [
    "_target_",
    "save_dir",
    "version",
    "offline",
    "dir",
    "id",
    "anonymous",
    "log_model",
    "experiment",
    "prefix",
    "checkpoint_name",
]

log = logging.getLogger(__name__)


def init_wandb(cfg: DictConfig) -> Run:
    logger_cfg = cfg.get("logger") or {}
    wandb_cfg = logger_cfg.get("wandb") or {}

    # set wandb to disabled if not using wandb logger
    use_wandb = isinstance(wandb_cfg, DictConfig) and "_target_" in wandb_cfg
    mode = wandb_cfg.get("mode") if use_wandb else "disabled"

    # remove notes from config before logging it to wandb, as it is not useful for sorting/filtering
    with open_dict(cfg):
        notes = wandb_cfg.pop("notes", None)

    # don't remove tags from config, as this enables sorting/filtering by tags

    # get kwargs for wandb.init() by removing keys that are only for Lightning's WandbLogger

    init_kwargs = {
        key: value for key, value in wandb_cfg.items() if key not in WANDB_LOGGER_KEYS
    }
    init_kwargs.update(
        {
            # mimic WandbLogger's treatment of save_dir/dir, version/id, and anonymous
            "dir": wandb_cfg.get("save_dir") or wandb_cfg.get("dir"),
            "id": wandb_cfg.get("version") or wandb_cfg.get("id"),
            "anonymous": "allow" if wandb_cfg.get("anonymous") else None,
            # over
            "mode": mode,
            "notes": notes,
        }
    )

    log.debug(
        "Initializing WandB run."
        if use_wandb
        else "WandB disabled. Initializing disabled dummy run."
    )

    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        **init_kwargs,  # type: ignore
    )
    return run
