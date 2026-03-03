from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from gymnasium.vector import VectorWrapper

from utils.trees import tree_call_method, tree_map

if TYPE_CHECKING:
    from gymnasium.vector import ActType, ArrayType, ObsType


class NumpyToTorch(VectorWrapper):
    """Wraps a numpy-based environment so that it can be interacted with
    through PyTorch Tensors.

    Based on Gymnasium's vectorized version of the ArrayConversion wrapper.
    Unfortunately, this wrapper uses the array_api_compat package which is not
    compatible with older versions of numpy.
    """

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        """Performs the given action within the environment.

        Args:
            actions: The actions to perform as any Array API compatible array

        Returns:
            The next observation, reward, termination, truncation, and extra info
        """
        # actions should already be on the cpu, as this is handled by the
        # GymEnvDataset
        actions = tree_call_method(actions, "numpy")
        obs, reward, terminated, truncated, info = self.env.step(actions)

        return (
            tree_map(torch.from_numpy, obs),
            tree_map(torch.from_numpy, reward),
            tree_map(torch.from_numpy, terminated),
            tree_map(torch.from_numpy, truncated),
            tree_map(torch.from_numpy, info),
        )

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment returning observation and info as Array from any Array API compatible framework.

        Args:
            seed: The seed for resetting the environment
            options: The options for resetting the environment, these are converted to jax arrays.

        Returns:
            xp-based observations and info
        """
        if options:
            # any arrays should already be on the same device as the environment
            options = tree_call_method(options, "numpy")

        obs, info = self.env.reset(seed=seed, options=options)
        return tree_map(torch.from_numpy, obs), tree_map(torch.from_numpy, info)
