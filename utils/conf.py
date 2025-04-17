from __future__ import annotations

import math
import os
import os.path as osp
from pathlib import Path
from typing import Sequence

import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict


def setup_resolvers(exclude: list[str] | None = None):
    exclude = exclude or []

    if "eval" not in exclude:
        OmegaConf.register_new_resolver("eval", eval)

    if "np" not in exclude:
        OmegaConf.register_new_resolver("np", lambda arr: np.array(arr))

    if "pi" not in exclude:
        OmegaConf.register_new_resolver("pi", lambda: np.pi)

    if "add" not in exclude:
        OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))

    if "prod" not in exclude:
        OmegaConf.register_new_resolver(
            "prod", lambda *numbers: np.prod(numbers).item()
        )

    if "floor_div" not in exclude:
        OmegaConf.register_new_resolver("floor_div", lambda x, y: x // y)

    if "sub" not in exclude:
        OmegaConf.register_new_resolver("sub", lambda x, y: x - y)

    if "log" not in exclude:
        OmegaConf.register_new_resolver("log", lambda x: math.log(x))

    if "abspath" not in exclude:
        OmegaConf.register_new_resolver("abspath", lambda s: osp.abspath(s))


def delete_keys_recursively(
    cfg: DictConfig, keys_to_delete: Sequence[str]
) -> DictConfig:
    with open_dict(cfg):
        for field in keys_to_delete:
            cfg.pop(field, None)

    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            cfg[key] = delete_keys_recursively(value, keys_to_delete)

    return cfg


def patch_load_from_checkpoint(
    cfg: DictConfig, checkpoint_path: os.PathLike | Path
) -> DictConfig:
    # instead of calling the class, call its `load_from_checkpoint()` method
    target = cfg["_target_"]
    target += ".load_from_checkpoint"
    cfg["_target_"] = target

    # prepend the checkpoint path to any positional arguments to
    # `load_from_checkpoint()`
    args = (checkpoint_path,)
    if "_args_" in cfg:
        args = args + cfg["_args_"]
    cfg["_args_"] = args

    return cfg
