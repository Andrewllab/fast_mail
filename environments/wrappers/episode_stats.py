from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch
from gymnasium.vector import VectorWrapper
from torch import Tensor

if TYPE_CHECKING:
    from gymnasium.vector import ActType, ArrayType, ObsType, VectorEnv

log = logging.getLogger(__name__)


class RecordEpisodeStatistics(VectorWrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: VectorEnv[ObsType, ActType, ArrayType],
        stats_key: str = "episode",
    ):
        gym.utils.RecordConstructorArgs.__init__(self, stats_key=stats_key)
        VectorWrapper.__init__(self, env)
        self.stats_key = stats_key
        self.stats = {}

        device = getattr(env.unwrapped, "device", torch.device("cpu"))
        self.episode_returns: Tensor = torch.zeros((self.num_envs,), device=device)
        self.episode_lengths: Tensor = torch.zeros(
            (self.num_envs,), dtype=torch.int, device=device
        )

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(seed=seed, options=options)

        if options is not None and "reset_mask" in options:
            reset_mask = options["reset_mask"]
            assert torch.is_tensor(
                reset_mask
            ), f"`options['reset_mask': mask]` must be a torch Tensor, got {type(reset_mask)}"
            assert reset_mask.shape == (
                self.num_envs,
            ), f"`options['reset_mask': mask]` must have shape `({self.num_envs},)`, got {reset_mask.shape}"
            assert (
                reset_mask.dtype == torch.bool
            ), f"`options['reset_mask': mask]` must have `dtype=torch.bool`, got {reset_mask.dtype}"
            assert torch.any(
                reset_mask
            ), f"`options['reset_mask': mask]` must contain a boolean array, got reset_mask={reset_mask}"

        else:
            reset_mask = Ellipsis

        self.episode_returns[reset_mask] = 0
        self.episode_lengths[reset_mask] = 0

        for key in self.stats.keys():
            self.stats[key][reset_mask] = 0

        # do not add episode statistics to info dict on reset, since we merge
        # the reset info dict with the step info dict in the ActionChunkWrapper,
        # and we don't want the episode statistics of completed trajectories
        # to be overwritten
        return obs, info

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(actions)

        self.episode_returns += rewards
        self.episode_lengths += 1

        return (
            observations,
            rewards,
            terminations,
            truncations,
            self.info(infos),
        )

    def info(self, infos: dict[str, Any]) -> dict[str, Any]:
        """Adds episode statistics to the info dict under the specified stats key."""

        if not isinstance(infos, dict):
            raise TypeError(
                f"`vector.RecordEpisodeStatistics` requires `info` type to be `dict`, its actual type is {type(infos)}. This may be due to usage of other wrappers in the wrong order."
            )

        for key, value in infos.items():
            if isinstance(value, dict):
                raise RuntimeError(
                    f"Nested info dicts are not supported by `RecordEpisodeStatistics`. Found nested dict at key '{key}' with value {value}."
                )

            if key not in self.stats:
                self.stats[key] = torch.zeros(
                    (self.num_envs,), dtype=value.dtype, device=value.device
                )

            if value.dtype == torch.bool:
                self.stats[key] = torch.logical_or(self.stats[key], value)
            else:
                self.stats[key] += value

        episode_info = {
            "return": self.episode_returns.clone(),
            "length": self.episode_lengths.clone(),
        }
        episode_info.update({key: stat.clone() for key, stat in self.stats.items()})

        infos[self.stats_key] = episode_info

        return infos
