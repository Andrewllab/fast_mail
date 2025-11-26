import functools
import logging
from typing import Optional

import gymnasium as gym
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def make(
    env_id: str,
    num_envs: int,
    obs_mode: str,
    control_mode: str,
    render_mode: Optional[str] = None,
    wrappers: Optional[DictConfig] = None,
    # max_episode_steps: int = 500,
    **kwargs,
):
    """
    Creates a ManiSkill environment and applies specified wrappers using Hydra.

    Args:
        env_id: The ID of the ManiSkill environment to create.
        num_envs: The number of parallel environments.
        obs_mode: The observation mode (e.g., 'rgbd', 'state').
        control_mode: The control mode for the robot.
        render_mode: The rendering mode.
        wrappers: A DictConfig from Hydra containing wrapper configurations.
                  Each wrapper must have '_target_' and '_partial_: True'.
        **kwargs: Additional keyword arguments for gym.make().
    """
    import mani_skill.envs

    log.info(f"Building ManiSkill env '{env_id}'...")

    env = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        # max_episode_steps=max_episode_steps,
        **kwargs,
    )

    # If no wrappers are defined in the config, return the base env.
    if wrappers is None:
        return env

    # Instantiate all wrappers defined in the config.
    # Because they are defined with '_partial_: True', this creates a
    # dictionary of "factory functions" waiting to be called.
    log.info("Instantiating wrappers...")
    wrapper_factories = hydra.utils.instantiate(wrappers)

    # Apply each wrapper to the environment.
    for wrapper_name, factory in wrapper_factories.items():
        if isinstance(factory, functools.partial):
            log.debug(f"Applying wrapper: {wrapper_name}")
            env = factory(env)
        else:
            log.warning(
                f"Item '{wrapper_name}' in wrappers config did not resolve to a partial function, skipping."
            )

    log.info("Finished applying all wrappers.")
    return env
