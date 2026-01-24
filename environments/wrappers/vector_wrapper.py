from __future__ import annotations

import logging
from typing import Any

import gymnasium.spaces as spaces
import torch
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.vector import VectorEnv

try:
    from gymnasium.vector import GymVectorWrapper
except ImportError:
    # backward compatibility for older gymnasium versions (0.29.1)
    from gymnasium.vector.vector_env import VectorEnvWrapper as GymVectorWrapper

log = logging.getLogger(__name__)


class VectorWrapper(GymVectorWrapper):

    # def get_wrapper_attr(self, name: str) -> Any:
    #     """Gets an attribute from the wrapper and lower environments if `name` doesn't exist in this object.

    #     Copied from gymnasium.core.Wrapper.

    #     Args:
    #         name: The variable name to get

    #     Returns:
    #         The variable with name in wrapper or lower environments
    #     """
    #     if hasattr(self, name):
    #         return getattr(self, name)
    #     else:
    #         try:
    #             return self.env.get_wrapper_attr(name)
    #         except AttributeError as e:
    #             raise AttributeError(
    #                 f"wrapper {self.__class__.__name__} has no attribute {name!r}"
    #             ) from e

    def render(self) -> tuple[RenderFrame, ...] | None:
        """Returns the render mode from the base vector environment.

        Copied from gymnasium.vector.VectorWrapper because it is missing in
        some older versions of gymnasium.
        """
        return self.env.render()
