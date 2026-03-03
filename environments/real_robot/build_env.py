import functools
import logging

import gymnasium as gym
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from environments.specs import DataSpecs
from environments.wrappers import ActionChunkWrapper, NumpyToTorch, RemoveInfoMasks

log = logging.getLogger(__name__)


def make_one(**kwargs) -> gym.Env:

    from .real_robot_env import RealRobotEnv

    env = RealRobotEnv(**kwargs)

    return env


def make(
    action_horizon: int | None = None,
    obs_seq_len: int = 1,
    **kwargs,
) -> gym.Env:
    env_fn = functools.partial(make_one, **kwargs)

    # create a vectorized wrapper around a single environment
    env = SyncVectorEnv(
        [env_fn],
        # do not copy observations since we don't modify them in-place anywhere
        copy=False,
        # all environments are the same, so they have the same observation space
        observation_mode="same",
        # we will handle auto-resetting ourselves in the ActionChunkWrapper
        autoreset_mode=AutoresetMode.DISABLED,
    )

    # Gymnasium's VectorEnv adds a mask for each field of the info dict to
    # indicate which envs' info dicts contains that field. Since all environments
    # are the same and always contain all fields, we can remove these.
    env = RemoveInfoMasks(env)

    # Convert numpy arrays to torch tensors, because all vectorized wrappers
    # expect torch tensors.
    env = NumpyToTorch(env)

    # We don't record episode statistics because the real environment doesn't
    # define a success condition

    # Handle action chunking and auto-resetting when any env is done.
    env = ActionChunkWrapper(
        env,
        action_horizon=action_horizon,
        obs_seq_len=obs_seq_len,
        auto_reset=True,  # reset env when done
    )

    # Get DataSpecs from underlying RealRobotEnv environment
    specs: DataSpecs = env.unwrapped.get_attr("specs")[0]

    # Adjust time properties of all specs according to obs and action sequence lengths
    specs = specs.set_obs_seq_len(obs_seq_len).set_action_seq_len(action_horizon)

    # assign as new attribute so that GymEnvDataset can access it
    env.specs = specs

    return env
