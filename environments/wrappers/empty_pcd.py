from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, SupportsFloat

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from gymnasium.core import ActType, ObsType

log = logging.getLogger(__name__)


class EmptyPointCloudChecker(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """
    A wrapper that checks if the point cloud observation is empty (i.e., has no valid points).
    If the point cloud is empty, the environment is reset.
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        min_points: int,
        max_depth: float,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self, min_points=min_points, max_depth=max_depth
        )
        gym.Wrapper.__init__(self, env)

        self.min_points = min_points
        self.max_depth = max_depth

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        n_points = 0
        for key, value in obs.items():
            if "depth" in key:
                n_points += (
                    np.logical_and(0 < value, value < self.max_depth).sum().item()
                )

        if n_points < self.min_points:
            log.warning(
                f"Fewer than {self.min_points} points are within max depth of {self.max_depth} ({n_points} points). Terminating episode."
            )
            terminated = True

        return obs, reward, terminated, truncated, info
