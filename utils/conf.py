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


def resolve_path(str_path: os.PathLike) -> Path:
    # resolve any environment variables in the first path element
    elements = str(str_path).split("/")
    # if first character of first element is "$"
    if (root := elements[0]).startswith("$"):
        elements[0] = os.environ[root[1:]]
    path = "/".join(elements)  # not osp.join because we used str.split above

    # resolve ~ to the user's home directory
    path = Path(path).expanduser()
    return path


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
