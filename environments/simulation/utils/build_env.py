import functools

import gymnasium as gym
import hydra
from omegaconf import DictConfig

from environments.simulation.utils.factories import launch_sim, resolve_env_cfg
from environments.simulation.utils.wrappers import (
    IsaacLabGymWrapper,
    IsaacLabPreProcess,
)


def make(
    task_name: str,
    device: str,
    num_envs: int,
    render_mode: str,
    cli_args: DictConfig | None = None,
    env_cfg: DictConfig | None = None,
    wrappers: DictConfig | None = None,
    # BackCompat: leave these two arguments, which are no longer used
    enabled_wrappers=None,
    wrapper_cfgs=None,
):

    app_launcher = launch_sim(cli_args)
    cfg = resolve_env_cfg(task_name, device, num_envs)

    if env_cfg is not None:
        # override default value in env configclass with values from the hydra config
        for key, value in env_cfg.items():
            setattr(cfg, str(key), value)

    env = gym.make(id=task_name, cfg=cfg, num_envs=num_envs, render_mode=render_mode)

    wrappers_partials = hydra.utils.instantiate(wrappers)

    if wrappers_partials is None:
        return env

    # filter out any values that are not partials
    wrappers_partials = [
        wrapper
        for wrapper in wrappers_partials.values()
        if isinstance(wrapper, functools.partial)
    ]

    # find the IsaacLabGymWrapper wrapper and move it to the front of the list
    # this must be the first wrapper so that it can add extra reset kwargs that
    # are not compatible with the gym.Wrapper API
    preprocessers = [w for w in wrappers_partials if w.func is IsaacLabGymWrapper]
    if len(preprocessers) != 1:
        raise ValueError("IsaacLab envs require exactly one IsaacLabGymWrapper wrapper")
    i = wrappers_partials.index(preprocessers[0])
    preprocess_wrapper = wrappers_partials.pop(i)
    wrappers_partials.insert(0, preprocess_wrapper)

    # find the IsaacLabPreProcess wrapper and move it to the end of the list
    # preprocess must be final wrapper so that GymEnvDataset can access the specs
    preprocessers = [w for w in wrappers_partials if w.func is IsaacLabPreProcess]
    if len(preprocessers) != 1:
        raise ValueError("IsaacLab envs require exactly one IsaacLabPreProcess wrapper")
    i = wrappers_partials.index(preprocessers[0])
    preprocess_wrapper = wrappers_partials.pop(i)
    wrappers_partials.append(preprocess_wrapper)

    for wrapper in wrappers_partials:
        env = wrapper(env)

    return env
