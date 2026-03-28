from __future__ import annotations

import logging
import re
from typing import Any

from omegaconf import DictConfig, open_dict

from transforms.base_transform import KEY_PATTERN as TRANSFORM_KEY_PATTERN
from transforms.base_transform import Sequential

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
                            "Applying custom callback `%s` to legacy target `%s`",
                            replace.__name__,
                            cfg._target_,
                        )
                        cfg = replace(cfg, match)
                    else:
                        raise ValueError(f"Unexpected replace type {type(replace)}")

    for key, value in cfg.items():
        if isinstance(value, DictConfig):
            cfg[key] = patch_legacy_configs(value)

    return cfg


def add_transform_prefixes(
    state_dict: dict[str, Any], transforms: Sequential
) -> dict[str, Any]:
    """The Sequential transform used to remove the numerical prefixes from
    transform names, e.g., t0_depth_remove_inf -> depth_remove_inf. This
    affects the keys in the state dict, which need to be updated accordingly.
    """

    transform_keys = list(transforms.keys())

    for transform_key in transform_keys:

        # find the legacy short name (without ordinal prefix) from the transform key
        # Sequential removes the t prefix from the transform keys, so we need
        # to add it back
        config_key = "t" + transform_key
        match = TRANSFORM_KEY_PATTERN.match(config_key)
        assert match is not None
        short_name = match.group(3)
        assert short_name is not None

        # create regex pattern to match state dict keys containing this
        # transform's short name
        # Regex pattern explanation:
        # (_ema_obs_encoder\.module\.)  - captures the fixed prefix.
        # re.escape(name)               - ensures that special characters in name are treated literally.
        # (\..*)                        - captures everything after the name (including the dot).
        pattern = rf"(_ema_obs_encoder\.module\.){re.escape(short_name)}(\..*)"
        # Regex pattern explanation:
        # Backreferences (\g<1>, \g<2>) - preserve the unchanged parts of the string.
        # Explicit backreferences (\g<1> instead of \1) avoids issues with
        # numbers in the transform key.
        replacement = rf"\g<1>{transform_key}\g<2>"

        for key in list(state_dict.keys()):
            if re.match(pattern, key):
                new_key = re.sub(pattern, replacement, key)
                log.debug(
                    "Replacing `%s` with `%s` in state dict key %s",
                    short_name,
                    transform_key,
                    key,
                )
                state_dict[new_key] = state_dict.pop(key)

    return state_dict


def rename_robocasa_cameras(state_dict: dict[str, Any]) -> dict[str, Any]:
    rename_mapping = {
        "left_cam": "robot0_agentview_left",
        "right_cam": "robot0_agentview_right",
        "gripper_cam": "robot0_eye_in_hand",
    }

    for old_name, new_name in rename_mapping.items():
        # Regex pattern explanation:
        # ^(_ema_obs_encoder\..*?\.RGBStream\.)     - Group 1:
        #                                               - literally "_ema_obs_encoder." at the start of the string
        #                                               - non-greedy match of any characters
        #                                               - until ".RGBStream."
        # old_name                                  - target text
        # (\..*)                                    - Group 2: The dot and everything after
        pattern = rf"^(_ema_obs_encoder\..*?\.RGBStream\.){old_name}(\..*)"
        replacement = rf"\g<1>{new_name}\g<2>"

        for key in list(state_dict.keys()):
            if re.match(pattern, key):
                new_key = re.sub(pattern, replacement, key)
                log.debug(
                    "Replacing `%s` with `%s` in state dict key %s",
                    old_name,
                    new_name,
                    key,
                )
                state_dict[new_key] = state_dict.pop(key)

    return state_dict


def patch_legacy_state_dict(
    state_dict: dict[str, Any], transforms: Sequential
) -> dict[str, Any]:
    """Patches legacy state dicts by updating keys to match the current
    model structure.

    Currently, this only involves adding transform name prefixes to the
    obs_encoder submodules in the state dict.
    """
    state_dict = add_transform_prefixes(state_dict, transforms)
    state_dict = rename_robocasa_cameras(state_dict)
    return state_dict
