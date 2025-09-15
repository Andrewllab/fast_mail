import functools
import logging

from typing import List
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# !!! IMPORTANT: the original create_env function was modified to take an additional parameter: camera_depths and then "pip install -e ." robocasa again !!!!
from robocasa.utils.env_utils import create_env


def make(
    env_name: str,
    img_size: List[int],
    camera_names: List[str],
    wrappers: DictConfig | None = None,
):
    """
    Creates a RoboCasa environment and applies specified wrappers using Hydra.

    Args:
        env_name: The environment name of the RoboCasa environment to create.
        img_size: Width and height of the observations.
        camera_names: List of camera names used for observations.
        wrappers: A DictConfig from Hydra containing wrapper configurations.
                  Each wrapper must have '_target_' and '_partial_: True'.
    """

    log.info(f"Building RoboCasa environment: '{env_name}'")

    env = create_env(
        env_name=env_name,
        camera_widths=img_size[0],
        camera_heights=img_size[1],
        camera_names=[
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ],
        camera_depths=True,
    )

    log.info("Instantiating wrappers...")
    wrappers_partials = hydra.utils.instantiate(wrappers)

    if wrappers_partials is None:
        return env

    # filter out any values that are not partials
    wrappers_partials = [
        wrapper
        for wrapper in wrappers_partials.values()
        if isinstance(wrapper, functools.partial)
    ]

    for wrapper in wrappers_partials:
        env = wrapper(env)

    log.info("Finished applying all wrappers.")

    return env

