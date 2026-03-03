from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import torch
from gymnasium.vector import VectorWrapper
from torch import Tensor

from utils.trees import tree_map

if TYPE_CHECKING:
    from gymnasium.vector import ActType, ArrayType, ObsType, VectorEnv


class DtypeObservation(VectorWrapper, gym.utils.RecordConstructorArgs):
    """Modifies the dtype of an observation array to a specified dtype.

    Note:
        This is only compatible with :class:`Dict` observation space.
    """

    def __init__(
        self,
        env: VectorEnv[ObsType, ActType, ArrayType],
        filter_dtype: str,
        target_dtype: str,
    ):
        """Wrapper class to change inputs and outputs of environment to any Array API framework.

        Args:
            env: The Array API compatible environment to wrap
            env_xp: The Array API framework the environment is on
            target_xp: The Array API framework to convert to
            env_device: The device the environment is on
            target_device: The device on which Arrays should be returned
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, filter_dtype=filter_dtype, target_dtype=target_dtype
        )
        VectorWrapper.__init__(self, env)
        self.filter_dtype = getattr(torch, filter_dtype)
        self.target_dtype = getattr(torch, target_dtype)

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options or {})
        return self.observations(obs), self.info(info)

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, Tensor, Tensor, Tensor, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return self.observations(obs), reward, terminated, truncated, self.info(info)

    def observations(self, observations: ObsType) -> ObsType:
        return tree_map(self._convert_dtype, observations)

    def info(self, info: dict[str, Any]) -> dict[str, Any]:
        return tree_map(self._convert_dtype, info)

    def _convert_dtype(self, array: Tensor) -> Tensor:
        if array.dtype == self.filter_dtype:
            return array.to(self.target_dtype)

        return array
