from __future__ import annotations

import os.path as osp

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

    if "sub" not in exclude:
        OmegaConf.register_new_resolver("sub", lambda x, y: x - y)

    if "abspath" not in exclude:
        OmegaConf.register_new_resolver("abspath", lambda s: osp.abspath(s))


def pop_names(cfg: DictConfig) -> DictConfig:
    with open_dict(cfg):
        cfg.pop("name", None)

    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            cfg[key] = pop_names(value)

    return cfg
