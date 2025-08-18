from omegaconf import DictConfig
from typing import Optional, List
import gymnasium as gym

from environments.simulation.utils.factories import launch_sim_and_prepare_env_cfg
from environments.simulation.utils.wrappers import IsaacLabPreProcess


def apply_wrappers(env, enabled_wrappers: List[str], wrapper_cfgs: DictConfig = None):

    for wrapper_name in enabled_wrappers:
        match wrapper_name:

            case "record_video":
                env = gym.wrappers.RecordVideo(env, **wrapper_cfgs["record_video"])

            case "pre_process":
                env = IsaacLabPreProcess(env, **wrapper_cfgs["pre_process"])

            case _:
                raise ValueError(
                    "Required wrapper is not supported: ",
                    wrapper_name,
                    ". Supported wrappers: [record_video, pre_process].",
                )

    return env


def make(
    task_name: str,
    device: str,
    num_envs: int,
    render_mode: str,
    cli_args: Optional[DictConfig],
    wrapper_cfgs: List[dict],
    enabled_wrappers: List[str],
):

    sim_launcher, base_env_cfg = launch_sim_and_prepare_env_cfg(
        task_name, device, num_envs, cli_args
    )

    env = gym.make(
        id=task_name, cfg=base_env_cfg, num_envs=num_envs, render_mode=render_mode
    )

    if wrapper_cfgs:
        env = apply_wrappers(env, enabled_wrappers, wrapper_cfgs)

    return env
