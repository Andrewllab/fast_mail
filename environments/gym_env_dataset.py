import logging
from typing import Literal, Mapping, Sequence, TypeVar

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
        obs, reset_info = self.env.reset()

        device = obs.device or "cpu"
        info = TensorDict(batch_size=(self.num_envs,), device=device)
        info = accumulate_dict(info, unnest_dict(reset_info), aggr="max")
        # TODO: move accumulation logic into a wrapper because it's
        # probably environment specific
        # TODO: also accumulate reward across episode

        # yield once with the reset observation
        # it's not possible that any envs are already done, so we don't yield info
        yield step_return_to_tensor_dict(obs)

        # afterwards, yield in a loop until we have reached the required number of episodes
        num_episodes = 0
        time = 0
        while True:
            # stop iterating if num_episodes is reached
            if self.num_episodes is not None and num_episodes >= self.num_episodes:
                break

            time += 1

            actions = self._next_action
            if actions is None:
                raise ValueError(
                    f"No actions received from agent on time step #{time}. Check that ActionWriter callback is setup properly."
                )
            self._next_action = None

            # obs is either a Tensor or a TensorDict, so either way supports `.device`
            reward = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
            episode_infos = []
            # actions: [num_envs, action_horizon, action_dim]
            # therefore we need to unbind the actions along the time dimension

            actions = actions.transpose(0, 1)[: self.action_horizon]
            for action in actions:
                if self.fps is not None:
                    self.clock.tick(self.fps)

                obs, step_reward, step_terminated, step_truncated, step_info = (
                    self.env.step(action)
                )

                # accumulate the return values over time
                # we drop all observations except the last, since stacking
                # observations is done by the FrameStackObservation wrapper
                reward += step_reward
                step_done = torch.logical_or(step_terminated, step_truncated)

                # accumulate the "max" of any success-like metrics
                info = accumulate_dict(info, unnest_dict(step_info), aggr="max")

                # check if any envs are done for this first time at this step
                if torch.logical_and(step_done, ~done).any():
                    # store the accumulated info for the envs that are done
                    # (indexing with a boolean tensor is always a copy)
                    # TODO: also store the total reward and length of the episode
                    episode_infos.append(info[step_done])

                done = torch.logical_or(done, step_done)

                # Break out of action loop if all environments are done
                if done.all():
                    break

            # reset the envs that are done
            if done.any():
                num_episodes += done.sum().item()
                log.debug(f"Completed {num_episodes} episodes.")

                # reset the info for the envs that are done
                # tensordict cannot handle info[done] = 0 because the fields
                # have different data types
                for value in info.values():
                    value[done] = 0

                obs, reset_info = self.env.reset(options={"mask": done})
                info = accumulate_dict(info, unnest_dict(reset_info), aggr="max")

            yield step_return_to_tensor_dict(obs, episode_infos)

    def write_actions(self, actions: torch.Tensor):
        if self._next_action is not None:
            log.warning(f"Overwriting unconsumed next action {self._next_action}.")
        self._next_action = actions

    def teardown(self) -> None:
        # reset next action between evaluation epochs
        self._next_action = None

    def close(self):
        self.env.close()


T = TypeVar("T", bound=Mapping)


def unnest_dict(d: T, parent_key: str = "", sep: str = "/") -> T:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if bool(parent_key) and bool(sep) else k
        if isinstance(v, Mapping):
            items.extend(unnest_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    # this weird syntax also works for TensorDict
    return type(d)(dict(items))


def accumulate_dict(d1, d2, aggr: Literal["sum", "max"] = "sum"):

    for key, value in d2.items():
        if key not in d1:
            d1[key] = value
        else:
            if aggr == "sum":
                d1[key] += value
            elif aggr == "max":
                d1[key] = max(d1[key], value)
            else:
                raise ValueError(f"Unknown aggregation method: {aggr}")

    return d1


def step_return_to_tensor_dict(
    obs: TensorDict,
    episode_infos: Sequence[TensorDict] | None = None,
) -> TensorDict:
    """Convert step return to TensorDict."""

    # obs should only have one leading dimension (batch, not time) resulting
    # from the vectorized environment stacking the observations from all envs
    assert obs.ndim == 1

    goal = obs.pop("goal", None)

    tensordict = TensorDict({"obs": obs})  # type: ignore

    # The final obs dict should have only a single leading dimension but we
    # also have to unsqueeze to add a singleton time dimension.
    # The only way we can unsqueeze the tensordict is like this:
    tensordict.auto_batch_size_(batch_dims=1)
    tensordict = tensordict.unsqueeze(dim=1)
    tensordict.auto_batch_size_(batch_dims=1)

    if goal is not None:
        tensordict["goal"] = goal

    if episode_infos:
        episode_info = tensordict.stack(episode_infos).float()
        tensordict["episode_info"] = episode_info

    return tensordict


def monkey_patch_data_fetcher():
    from lightning.pytorch.loops.fetchers import _PrefetchDataFetcher

    class _NoPrefetchDataFetcher(_PrefetchDataFetcher):
        def __init__(self, prefetch_batches: int = 0) -> None:
            super().__init__(prefetch_batches=prefetch_batches)

    import lightning.pytorch.loops.utilities as utilities

    utilities._PrefetchDataFetcher = _NoPrefetchDataFetcher
