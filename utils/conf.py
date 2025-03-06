from __future__ import annotations

import os.path as osp

import numpy as np
from omegaconf import OmegaConf


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
