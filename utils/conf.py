from __future__ import annotations

import logging
import math
import os
import os.path as osp
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict

log = logging.getLogger(__name__)


def if_resolver(pred: bool, a, b):
    chosen = a if pred else b
    return OmegaConf.create(chosen) if isinstance(chosen, (dict, list)) else chosen


def setup_resolvers(exclude: list[str] | None = None):
    exclude = exclude or []

    if "eval" not in exclude:
        OmegaConf.register_new_resolver("eval", eval)

    if "if" not in exclude:
        OmegaConf.register_new_resolver("if", if_resolver)

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

    if "pow" not in exclude:
        OmegaConf.register_new_resolver("pow", lambda x, y: math.pow(x, y))

    if "abspath" not in exclude:
        OmegaConf.register_new_resolver("abspath", lambda s: osp.abspath(s))

    if "join" not in exclude:
        OmegaConf.register_new_resolver("join", lambda *args: osp.join(*args))

    if "dirname" not in exclude:
        OmegaConf.register_new_resolver("dirname", lambda s: osp.dirname(s))

    if "pathname" not in exclude:
        OmegaConf.register_new_resolver("pathname", lambda s: osp.basename(s))

    if "append_to_stem" not in exclude:
        OmegaConf.register_new_resolver(
            "append_to_stem",
            lambda path, suffix: str(Path(path).with_stem(Path(path).stem + suffix)),
        )


def log_slurm_job_id(cfg: DictConfig):
    """Add slurm job ID (and potentially the array job/task ID) to the cfg
    (under cfg.platform) where it will be logged to wandb.
    """
    with open_dict(cfg.platform):
        if "SLURM_ARRAY_TASK_ID" in os.environ:
            # if the job is part of an array job, it also has a task ID
            cfg.platform.slurm_array_task_id = os.environ["SLURM_ARRAY_TASK_ID"]
            cfg.platform.slurm_job_id = os.environ["SLURM_ARRAY_JOB_ID"]
        elif "SLURM_JOB_ID" in os.environ:
            cfg.platform.slurm_job_id = os.environ["SLURM_JOB_ID"]


def delete_keys_recursively(
    cfg: DictConfig, keys_to_delete: Sequence[str]
) -> DictConfig:
    with open_dict(cfg):
        for field in keys_to_delete:
            cfg.pop(field, None)

    for value in cfg.values():
        if isinstance(value, DictConfig):
            delete_keys_recursively(value, keys_to_delete)

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


def merge_data_configs(cfg: DictConfig, other: DictConfig) -> DictConfig:

    obj_keys = ["dataset", "env"]

    for key in obj_keys:
        if key not in other:
            # nothing to merge
            continue

        subcfg = other[key]
        subkeys = flatten_keys(subcfg)

        if "_target_" in subkeys or key not in cfg:
            # If the other subcfg contains a key anywhere called "_target_", then
            # this entire subconfig has been overridden. Replace value cfg with
            # value in other
            cfg[key] = subcfg

        else:
            # Only specific hyperparameters have been overwritten, so use the
            # standard merge algorithm. This keeps anything in cfg that isn't
            # explicitly overwritten
            cfg[key] = OmegaConf.merge(cfg[key], subcfg)

    transform_keys = [
        "cpu_transforms",
        "cpu_batch_transforms",
        "gpu_batch_transforms",
    ] + [k for k in other.keys() if "process" in k]

    for key in transform_keys:
        if key not in other:
            continue

        subcfg = other[key]
        subkeys = flatten_keys(subcfg)

        # # TODO: better algorithm for merging transform configs
        # # TODO: is there any way to detect if transforms should be replaced or merged?
        # if "_target_" in subkeys or key not in cfg:
        #     # If the other subcfg contains a key anywhere called "_target_", then
        #     # this entire subconfig has been overridden. Replace value cfg with
        #     # value in other
        #     cfg[key] = subcfg

        # else:
        #     # Only specific hyperparameters have been overwritten, so use the
        #     # standard merge algorithm. This keeps anything in cfg that isn't
        #     # explicitly overwritten
        #     cfg[key] = OmegaConf.merge(cfg[key], subcfg)

        cfg[key] = OmegaConf.merge(cfg[key], subcfg)

    primitive_keys = [
        k for k in other.keys() if k not in obj_keys and k not in transform_keys
    ]

    primitive_cfg = OmegaConf.masked_copy(other, primitive_keys)
    cfg = OmegaConf.merge(cfg, primitive_cfg)

    return cfg


def flatten_keys(cfg: Mapping[str, Any]) -> list[str]:
    keys = []
    for key, value in cfg.items():
        if isinstance(value, Mapping):
            keys.extend(flatten_keys(value))
        else:
            keys.append(key)

    return keys
