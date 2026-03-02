from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import tensordict
import torch
from tensordict import NonTensorData, TensorDict
from torch.utils.data import IterableDataset

from environments.specs import DataSpecs

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv

    from environments.wrappers import RecordMultiEpisodeVideo

log = logging.getLogger(__name__)


class DoneEvalSignal(BaseException):
    pass


class GymEnvDataset(IterableDataset):
    def __init__(
        self,
        env: VectorEnv,
        num_episodes: int | None = 20,
    ):
        self.env = env
        self.num_episodes = num_episodes

        self._specs = env.specs
        self.num_envs = env.unwrapped.num_envs
        self.device = getattr(env.unwrapped, "device", torch.device("cpu"))

        self._next_action = None

        # ensure that the DataLoader does not prefetch any samples, since they
        # don't exist until the agent provides actions
        monkey_patch_data_fetcher()

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    @property
    def video_recorder(self) -> RecordMultiEpisodeVideo | None:
        from gymnasium.vector import VectorWrapper

        from environments.wrappers import RecordMultiEpisodeVideo

        env = self.env
        video_recorders = []
        while isinstance(env, VectorWrapper):
            if isinstance(env, RecordMultiEpisodeVideo):
                video_recorders.append(env)

            env = env.env

        if len(video_recorders) > 1:
            log.warning(
                f"Multiple RecordMultiEpisodeVideo wrappers found in env: {video_recorders}. Returning the first one."
            )
            return video_recorders[0]
        elif len(video_recorders) == 1:
            return video_recorders[0]
        else:
            return None

    def __iter__(self):
        obs, reset_info = self.env.reset()

        # yield once with the reset observation
        yield self._step_return_to_tensor_dict(obs, reset_info)

        # afterwards, yield in a loop until we have reached the required number of episodes
        num_episodes = 0
        num_successes = 0
        time = 0
        while True:
            # stop iterating if num_episodes is reached
            if self.num_episodes is not None and num_episodes >= self.num_episodes:
                log.debug(f"Reached target of {self.num_episodes} episodes.")
                break

            time += 1

            action = self._next_action
            if action is None:
                raise ValueError(
                    f"No actions received from agent on time step #{time}. Check that ActionWriter callback is setup properly."
                )
            self._next_action = None

            # if the environment is on the cpu, copy all actions to the cpu
            # now instead of copying each action in the chunk separately
            action = action.to(device=self.device)

            try:
                obs, reward, terminated, truncated, info = self.env.step(action)

            except DoneEvalSignal:
                log.info("Aborting evaluation due to DoneEvalSignal.")
                # Return immediately without yielding next observation and info,
                # since we assume this signal is used during interactive
                # evaluation in the real world.
                # Do not reset the environment, since it pauses for user input.
                return

            # reset the envs that are done
            done = torch.logical_or(terminated, truncated)
            if done.any():
                num_episodes += done.sum().item()
                if "episode" in info and "success" in info["episode"]:
                    num_successes += info["episode"]["success"][done].sum().item()
                    log.debug(
                        f"Completed {num_episodes} episodes ({num_successes} successful)."
                    )
                else:
                    log.debug(f"Completed {num_episodes} episodes.")

            # yield even if num_episodes is reached so that the success/metrics
            # of the final episode are also logged by the agent
            yield self._step_return_to_tensor_dict(obs, info, reward=reward, done=done)

        # TODO: flush video recorder here

    def write_actions(self, actions: torch.Tensor):
        if self._next_action is not None:
            log.warning(f"Overwriting unconsumed next action {self._next_action}.")
        self._next_action = actions

    def teardown(self) -> None:
        # reset next action between evaluation epochs
        self._next_action = None

    def close(self):
        self.env.close()

    def _step_return_to_tensor_dict(
        self,
        obs: dict[str, Any],
        info: dict[str, Any],
        reward: torch.Tensor | None = None,
        done: torch.Tensor | None = None,
    ) -> TensorDict:
        """Convert step return to TensorDict."""

        if reward is None:
            reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        if done is None:
            done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Copy collate behavior in training to ensure consistency.
        # Reference: environments.collate.collate_tensor_dict
        with tensordict.set_list_to_stack(False):

            td = TensorDict({"obs": obs}, batch_size=self.num_envs, device=self.device)

            if "goal" in info:
                td["goal"] = info.pop("goal")

            if "episode" in info:
                td["episode_info"] = info.pop("episode")

            td["reward"] = reward
            td["done"] = done

            td["info"] = TensorDict(info, batch_size=self.num_envs, device=self.device)

        return td


def monkey_patch_data_fetcher():
    from lightning.pytorch.loops.fetchers import _PrefetchDataFetcher

    class _NoPrefetchDataFetcher(_PrefetchDataFetcher):
        def __init__(self, prefetch_batches: int = 0) -> None:
            super().__init__(prefetch_batches=prefetch_batches)

    import lightning.pytorch.loops.utilities as utilities

    utilities._PrefetchDataFetcher = _NoPrefetchDataFetcher
