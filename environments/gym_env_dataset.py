import dataclasses
import functools
import logging
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from tensordict import TensorDict
from torch.utils.data import IterableDataset

from environments.specs import DataSpecs

log = logging.getLogger(__name__)


class GymEnvDataset(IterableDataset):
    def __init__(
        self,
        env: Callable[[], gym.Env],
        num_envs: int = 1,
        num_episodes: int | None = 10,
        obs_seq_len: int = 1,
        action_horizon: int | None = None,
    ):
        if not isinstance(env, functools.partial):
            raise ValueError(
                "GymEnvDataset requires a callable that returns a gym.Env instance. Set _partial_ to True in the env config."
            )

        # we need to disable automatic resets, since the agent predicts action
        # sequences
        self.env = SyncVectorEnv(
            [env] * num_envs, copy=False, autoreset_mode=AutoresetMode.DISABLED
        )

        try:
            # VecEnvs return a tuple of results whenever an attribute is accessed
            one_step_specs: DataSpecs = self.env.get_attr("specs")[0]

        except AttributeError:
            log.error("Gym environment does not define specs.")
            raise

        obs = dict(one_step_specs.obs)  # copy obs specs for local modification
        for key, spec in obs.items():
            obs[key] = dataclasses.replace(spec, shape=(action_horizon, *spec.shape))
        action = dataclasses.replace(
            one_step_specs.action,
            shape=(action_horizon, *one_step_specs.action.shape),
        )
        self._specs = one_step_specs.replace(
            obs=obs,
            action=action,
        )

        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.action_horizon = action_horizon
        # TODO: implement obs_seq_len using FrameStackObservation. This must be
        # applied within the vec env, since each env is reset independently.

        self._next_action = None

        monkey_patch_data_fetcher()

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __iter__(self):
        # yield once with the reset observation
        obs, info = self.env.reset()
        yield step_return_to_tensor_dict(obs, info)

        # afterwards, yield in a loop until we have reached the required number of episodes
        num_episodes = 0
        time = 0
        while True:
            # stop iterating if num_episodes is reached
            if self.num_episodes is not None and num_episodes >= self.num_episodes:
                break

            actions = self._next_action
            if actions is None:
                raise ValueError(
                    f"No actions received from agent on time step #{time}. Check that ActionWriter callback is setup properly."
                )
            self._next_action = None

            reward = np.zeros(self.num_envs, dtype=np.float32)
            terminated = np.zeros(self.num_envs, dtype=bool)
            truncated = np.zeros(self.num_envs, dtype=bool)
            infos = []
            # actions: [num_envs, action_horizon, action_dim]
            # therefore we need to unbind the actions along the time dimension

            actions = actions.transpose(0, 1)[: self.action_horizon]
            actions_np = actions.cpu().numpy()
            for action in actions_np:
                obs, step_reward, step_terminated, step_truncated, step_info = (
                    self.env.step(action)
                )

                # accumulate the return values over time
                # we only ever need the last observation, since stacking
                # observations is done by the FrameStackObservation wrapper
                reward += step_reward
                terminated = np.logical_or(terminated, step_terminated)
                truncated = np.logical_or(truncated, step_truncated)
                infos.append(step_info)

            time += 1

            # we stack in axis=1 because the first dimension is the batch,
            # and the second dimension is the time dimension
            info = {
                key: np.stack([info[key] for info in infos], axis=1) for key in infos[0]
            }

            # reset the envs that are done
            done = np.logical_or(terminated, truncated)
            num_episodes += np.sum(done)
            if done.any():
                obs, reset_info = self.env.reset(options={"mask": done})
                info |= reset_info

            yield step_return_to_tensor_dict(
                obs,
                info,
                reward=reward,
                done=done,
            )

    def write_actions(self, actions: torch.Tensor):
        if self._next_action is not None:
            log.warning(f"Overwriting unconsumed next action {self._next_action}.")
        self._next_action = actions

    def close(self):
        self.env.close()


def step_return_to_tensor_dict(
    obs: np.ndarray,
    info: dict,
    reward: np.ndarray | None = None,
    done: np.ndarray | None = None,
) -> TensorDict:
    """Convert step return to TensorDict."""
    tensordict = TensorDict({"obs": obs})  # type: ignore
    tensordict.auto_batch_size_(batch_dims=1)

    if reward is None:
        reward = np.zeros(tensordict.shape[0], dtype=np.float32)
    if done is None:
        done = np.zeros(tensordict.shape[0], dtype=bool)

    success = info.pop("success", None) or info.pop("is_success", None)
    if success is None:
        success = np.zeros(tensordict.shape[0], dtype=bool)

    info["reward"] = reward
    tensordict.update(
        {
            "done": done,
            "success": success,
            "info": info,
        },  # type: ignore
    )
    # unsqueeze to add singleton time dimension
    return tensordict.unsqueeze(dim=0)


def monkey_patch_data_fetcher():
    from lightning.pytorch.loops.fetchers import _PrefetchDataFetcher

    class _NoPrefetchDataFetcher(_PrefetchDataFetcher):
        def __init__(self, prefetch_batches: int = 0) -> None:
            super().__init__(prefetch_batches=prefetch_batches)

    import lightning.pytorch.loops.utilities as utilities

    utilities._PrefetchDataFetcher = _NoPrefetchDataFetcher
