from __future__ import annotations

import logging
import os
import os.path as osp
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict

log = logging.getLogger(__name__)


def update_gc_samplers(cfg, match):
    cfg.pop("disable", None)
    return cfg


CONFIG_CALLBACKS = {
    ## Agents
    "agents.backbones.sigma_encoder.DDPM_SigmaEncoder": "models.sigma_encoders.DDPMSigmaEncoder",
    # The only difference between BESO_SigmaEncoder and DDPM_SigmaEncoder was
    # the replacement of sigma by log(sigma)/4 in the forward pass. This has
    # now been subsumed into the BesoAgent's edm_preconditioning method, so we
    # can alias the two classes.
    "agents.backbones.sigma_encoder.BESO_SigmaEncoder": "models.sigma_encoders.DDPMSigmaEncoder",
    "agents.edm_diffusion.gc_sampling.sample_.*": update_gc_samplers,
    ## Models
    "models.pos_encoder.SinusoidalTokenPosEncoder": "models.pos_encoder.SinusoidalSequencePosEncoder",
    ## Transforms
    "transforms.action_abs_to_relative.AbsoluteActionToRelativeChunk": "transforms.action_abs_to_relative.AbsoluteEeActionsToRelativeChunk",
    # the following 3 jitter variants were merged into a common Jitter transform
    "transforms.pointcloud_jitter_points.JitterPointCloud": "transforms.jitter.JitterPointCloud",
    "transforms.translational_jitter.TranslationalJitter": "transforms.jitter.TranslationalJitter",
    "transforms.translational_jitter.VariableTranslationalJitter": "transforms.jitter.VariableTranslationalJitter",
    "transforms.time_trim_idle.TrimIdleStart": "transforms.time_trim_idle.TrimIdle",
}

CONFIG_CALLBACKS = {re.compile(key): value for key, value in CONFIG_CALLBACKS.items()}


def patch_legacy_configs(cfg: DictConfig) -> DictConfig:
    """Patches legacy configs by replacing _target_ class paths with their
    updated paths or applying callbacks to update the config.

    Uses the CONFIG_CALLBACKS mapping defined at the top of this file.
    """
    if "_target_" in cfg:
        for pattern, replace in CONFIG_CALLBACKS.items():
            if match := pattern.fullmatch(cfg._target_):
                with open_dict(cfg):
                    if isinstance(replace, str):
                        log.info(
                            "Replacing legacy target `%s` with `%s`",
                            cfg._target_,
                            replace,
                        )
                        cfg._target_ = replace
                    elif callable(replace):
                        log.info(
                            "Applying custom callback to legacy target `%s`",
                            cfg._target_,
                        )
                        cfg = replace(cfg, match)
                    else:
                        raise ValueError(f"Unexpected replace type {type(replace)}")

    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            cfg[key] = patch_legacy_configs(value)

    return cfg
