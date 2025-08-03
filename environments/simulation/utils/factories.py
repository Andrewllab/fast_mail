from omegaconf import DictConfig
from typing import Optional

from isaaclab.app import AppLauncher


def env_cfg_factory(
    task_name: str, device: str, num_envs: int, cli_args: Optional[DictConfig] = None
):

    # Launch Isaac Sim through the AppLauncher simulation application
    # One could split the simulation start into a separate function and then call it, but making a separate function for one line is over-engineering.
    app_launcher = AppLauncher(cli_args)

    # Import necessary isaac-modules only after starting the simulators since various dependency modules of Isaac Sim are only available after the simulation app is running
    # For more details: https://isaac-sim.github.io/IsaacLab/main/source/tutorials/00_sim/create_empty.html
    from isaaclab_tasks.utils import parse_env_cfg

    # Import custom environments
    import environments.simulation.furniture_bench

    # parse/prepare the environment configuration for gym.make
    # https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/utils/parse_cfg.py
    return parse_env_cfg(
        task_name=task_name,
        device=device,
        num_envs=num_envs,
    )
