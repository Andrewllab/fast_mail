import torch
from gymnasium.logger import warn
from gymnasium.vector import AutoresetMode, VectorEnv, VectorWrapper
from tensordict import TensorDict


class VectorToTorchWrapper(VectorWrapper):
    """
    A wrapper that converts the observations from a vectorized environment to PyTorch tensors.
    """

    def __init__(self, env: VectorEnv):
        """Vector observation wrapper that batch transforms observations.

        Args:
            env: Vector environment.
        """
        super().__init__(env)
        if "autoreset_mode" not in env.metadata:
            warn(
                f"Vector environment ({env}) is missing `autoreset_mode` metadata key."
            )
        else:
            assert (
                env.metadata["autoreset_mode"] == AutoresetMode.NEXT_STEP
                or env.metadata["autoreset_mode"] == AutoresetMode.DISABLED
            )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._convert(obs), self._convert(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return (
            self._convert(obs),
            self._convert(reward),
            self._convert(terminated),
            self._convert(truncated),
            self._convert(info),
        )

    def _convert(self, value):
        if isinstance(value, dict):
            return TensorDict(value, batch_size=self.num_envs)
        else:
            return torch.from_numpy(value)
