from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gymnasium.vector import VectorWrapper

if TYPE_CHECKING:
    from gymnasium.vector import ActType, ArrayType, ObsType


class RemoveInfoMasks(VectorWrapper):
    """Removes masks from the info dictionary added by Gymnasium's VectorEnv.
    For each key in the info dict, VectorEnv adds a corresponding _key field,
    which is a boolean mask indicating which environments have this key.
    """

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options or {})
        return obs, self.info(info)

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(actions)
        return obs, reward, terminated, truncated, self.info(info)

    def info(self, info: dict[str, Any]) -> dict[str, Any]:
        for key in list(info.keys()):
            if isinstance(info[key], dict):
                info[key] = self.info(info[key])

            if key.startswith("_") and key[1:] in info:
                del info[key]

        return info
