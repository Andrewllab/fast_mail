import functools
import logging

import gymnasium as gym
import hydra
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from omegaconf import DictConfig

from environments.specs import DataSpecs

log = logging.getLogger(__name__)


def make(*args, wrappers: DictConfig | None = None, **kwargs) -> gym.Env:
    """
    Creates a RoboCasa environment and applies specified wrappers using Hydra.

    Args:
        env_name: The environment name of the RoboCasa environment to create.
        img_size: Width and height of the observations.
        camera_names: List of camera names used for observations.
        wrappers: A DictConfig from Hydra containing wrapper configurations.
                  Each wrapper must have '_target_' and '_partial_: True'.
    """

    from environments.wrappers import VectorToTorchWrapper

    from .real_robot_env import RealRobotEnv

    wrappers_partials = hydra.utils.instantiate(wrappers)
    if wrappers_partials is not None:
        # filter out any values that are not partials
        wrappers_partials = [
            wrapper
            for wrapper in wrappers_partials.values()
            if isinstance(wrapper, functools.partial)
        ]

    def make_one() -> gym.Env:

        env = RealRobotEnv(*args, **kwargs)

        if wrappers_partials is not None:
            log.info("Instantiating wrappers...")
            for wrapper in wrappers_partials:
                env = wrapper(env)

            log.info("Finished applying all wrappers.")

        return env

    # we need to disable automatic resets, since the agent predicts action
    # sequences
    env = SyncVectorEnv([make_one], copy=False, autoreset_mode=AutoresetMode.DISABLED)

    # gymnasium's VectorEnv converts Tensors to numpy arrays, so we need to
    # convert them back to Tensors
    env = VectorToTorchWrapper(env)

    # VecEnvs return a tuple of results whenever an attribute is accessed
    one_step_specs: DataSpecs = env.unwrapped.get_attr("specs")[0]

    # assign as new attribute so that GymEnvDataset can access it
    env.specs = one_step_specs

    return env
