import logging
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Space
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
        # we need to disable automatic resets, since the agent predicts action
        # sequences
        self.env = SyncVectorEnv(
            [env] * num_envs, copy=False, autoreset_mode=AutoresetMode.DISABLED
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
        try:
            # VecEnvs return a tuple of results whenever an attribute is accessed
            return self.env.get_attr("specs")[0]
        except AttributeError:
            return spaces_to_specs(self.env.observation_space, self.env.action_space)

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
            for action in actions[:, : self.action_horizon].unbind(dim=1):
                obs, step_reward, step_terminated, step_truncated, step_info = (
                    self.env.step(action.cpu().numpy())
                )

                # accumulate the return values over time
                # we only ever need the last observation, since stacking
                # observations is done by the FrameStackObservation wrapper
                reward += step_reward
                terminated = np.logical_or(terminated, step_terminated)
                truncated = np.logical_or(truncated, step_truncated)
                infos.append(step_info)

            time += 1

            # reset the envs that are done
            done = np.logical_or(terminated, truncated)
            num_episodes += np.sum(done)
            if done.any():
                obs = self.env.reset(options={"mask": done})

            # we stack in axis=1 because the first dimension is the batch,
            # and the second dimension is the time dimension
            info = {
                key: np.stack([info[key] for info in infos], axis=1) for key in infos[0]
            }
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
    return tensordict


def spaces_to_specs(obs_space: Space, action_space: Space) -> DataSpecs:
    # TODO: write function that either converts gym spaces to specs or vice versa
    raise NotImplementedError


def monkey_patch_data_fetcher():
    from lightning.pytorch.loops.fetchers import (
        _DataFetcher,
        _DataLoaderIterDataFetcher,
        _PrefetchDataFetcher,
    )
    from lightning.pytorch.trainer.states import RunningStage
    from lightning.pytorch.utilities.rank_zero import rank_zero_warn
    from lightning.pytorch.utilities.signature_utils import is_param_in_hook_signature

    def _select_data_fetcher(
        trainer: "pl.Trainer", stage: RunningStage
    ) -> _DataFetcher:
        lightning_module = trainer.lightning_module
        if stage == RunningStage.TESTING:
            step_fx_name = "test_step"
        elif stage == RunningStage.TRAINING:
            step_fx_name = "training_step"
        elif stage in (RunningStage.VALIDATING, RunningStage.SANITY_CHECKING):
            step_fx_name = "validation_step"
        elif stage == RunningStage.PREDICTING:
            step_fx_name = "predict_step"
        else:
            raise RuntimeError(f"DataFetcher is unsupported for {trainer.state.stage}")
        step_fx = getattr(lightning_module, step_fx_name)
        if is_param_in_hook_signature(step_fx, "dataloader_iter", explicit=True):
            rank_zero_warn(
                f"Found `dataloader_iter` argument in the `{step_fx_name}`. Note that the support for "
                "this signature is experimental and the behavior is subject to change."
            )
            return _DataLoaderIterDataFetcher()
        return _PrefetchDataFetcher(prefetch_batches=0)

    import lightning.pytorch.loops.evaluation_loop as evaluation_loop
    import lightning.pytorch.loops.prediction_loop as prediction_loop

    prediction_loop._select_data_fetcher = _select_data_fetcher
    evaluation_loop._select_data_fetcher = _select_data_fetcher
