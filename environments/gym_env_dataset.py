import logging

import gymnasium as gym
import pygame
import torch
from tensordict import TensorDict
from torch.utils.data import IterableDataset

from environments.specs import DataSpecs

log = logging.getLogger(__name__)


class GymEnvDataset(IterableDataset):
    def __init__(
        self,
        env: gym.Env,
        num_envs: int = 1,
        num_episodes: int | None = 10,
        obs_seq_len: int = 1,
        action_horizon: int | None = None,
        fps: float | None = None,
    ):
        self.env = env

        one_step_specs: DataSpecs = env.specs
        specs = one_step_specs.set_obs_seq_len(obs_seq_len)
        # if the action horizon is None, all predicted actions are executed
        specs = specs.set_action_seq_len(action_horizon if action_horizon else 1)
        self._specs = specs

        # TODO: infer from data
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.action_horizon = action_horizon
        # TODO: implement obs_seq_len using FrameStackObservation. This must be
        # applied within the vec env, since each env is reset independently.

        self.clock = pygame.time.Clock()
        self.fps = fps

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

            # obs is either a Tensor or a TensorDict, so either way supports `.device`
            device = obs.device or "cpu"
            reward = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
            terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            # actions: [num_envs, action_horizon, action_dim]
            # therefore we need to unbind the actions along the time dimension

            actions = actions.transpose(0, 1)[: self.action_horizon]
            for action in actions:
                if self.fps is not None:
                    self.clock.tick(self.fps)

                obs, step_reward, step_terminated, step_truncated, info = self.env.step(
                    action
                )

                # accumulate the return values over time
                # we drop all observations except the last, since stacking
                # observations is done by the FrameStackObservation wrapper
                # we also only keep the last info dict, mainly because there
                # is no clear way to accumulate the information over time
                # while also handling the reset info
                reward += step_reward
                terminated = torch.logical_or(terminated, step_terminated)
                truncated = torch.logical_or(truncated, step_truncated)

            time += 1

            # reset the envs that are done
            done = torch.logical_or(terminated, truncated)
            num_episodes += done.sum().item()
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
    obs: TensorDict,
    info: TensorDict,
    reward: torch.Tensor | None = None,
    done: torch.Tensor | None = None,
) -> TensorDict:
    """Convert step return to TensorDict."""

    # obs and info should only have one leading dimension corresponding to
    # the batch resulting from the vectorized environment stacking the
    # observations from all envs
    assert obs.ndim == 1
    assert info.ndim == 1
    assert obs.shape == info.shape

    # reward, done, and success
    if reward is None:
        reward = torch.zeros(obs.shape, dtype=torch.float32)
    if done is None:
        done = torch.zeros(obs.shape, dtype=torch.bool)

    for key in ("success", "is_success", "Episode_Termination/success"):
        if key in info:
            success = info.pop(key)
            break
    else:
        success = torch.zeros(obs.shape, dtype=torch.bool)

    tensordict = TensorDict(
        {
            "obs": obs,
            "done": done,
            "success": success,
            "reward": reward,
            "info": info,
        }  # type: ignore
    )

    # the final batch should have only a single leading dimension for the batch
    # but we also have to unsqueeze to add a singleton time dimension
    # the only way we can unsqueeze the tensordict is like this:
    tensordict.auto_batch_size_(batch_dims=1)
    tensordict = tensordict.unsqueeze(dim=1)
    tensordict.auto_batch_size_(batch_dims=1)

    return tensordict


def monkey_patch_data_fetcher():
    from lightning.pytorch.loops.fetchers import _PrefetchDataFetcher

    class _NoPrefetchDataFetcher(_PrefetchDataFetcher):
        def __init__(self, prefetch_batches: int = 0) -> None:
            super().__init__(prefetch_batches=prefetch_batches)

    import lightning.pytorch.loops.utilities as utilities

    utilities._PrefetchDataFetcher = _NoPrefetchDataFetcher
