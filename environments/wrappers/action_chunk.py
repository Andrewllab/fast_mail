from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import gymnasium.spaces as spaces
import torch
from gymnasium.vector import VectorWrapper
from gymnasium.vector.utils import batch_space
from torch import Tensor

from utils.trees import tree_get_item

if TYPE_CHECKING:
    from gymnasium.vector import VectorEnv

log = logging.getLogger(__name__)


class ActionChunkWrapper(VectorWrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: VectorEnv[dict[str, Tensor], Tensor, Tensor],
        action_horizon: int | None = None,
        obs_seq_len: int = 1,
        auto_reset: bool = False,
        use_render_obs: bool = False,
    ):
        # TODO: implement obs_seq_len. We cannot use FrameStackObservation here
        # because part of the observation comes from render(), which is not
        # handled by FrameStackObservation.
        if action_horizon is not None and action_horizon <= 0:
            raise ValueError("action_horizon must be a positive integer")

        if obs_seq_len != 1:
            raise NotImplementedError("obs_seq_len > 1 is not yet implemented")

        if use_render_obs:
            raise NotImplementedError("use_render_obs is not yet implemented")

        gym.utils.RecordConstructorArgs.__init__(
            self,
            action_horizon=action_horizon,
            obs_seq_len=obs_seq_len,
            auto_reset=auto_reset,
            use_render_obs=use_render_obs,
        )
        VectorWrapper.__init__(self, env)
        self.action_horizon = action_horizon
        self.auto_reset = auto_reset
        self.obs_seq_len = obs_seq_len
        self.use_render_obs = use_render_obs

        if self.use_render_obs and self.has_wrapper_attr("render_obs_space"):
            if not isinstance(self.observation_space, spaces.Dict):
                raise ValueError(
                    "ActionChunkWrapper can only be used with render observations if the observation space is a Dict."
                )

            render_obs_space = self.get_wrapper_attr("render_obs_space")
            if not isinstance(render_obs_space, spaces.Dict):
                raise ValueError(
                    "ActionChunkWrapper can only be used with render observations if the render observation space is a Dict."
                )

            self.observation_space = spaces.Dict(
                {
                    **self.observation_space.spaces,
                    **render_obs_space.spaces,
                }
            )

        self.single_observation_space = batch_space(
            env.single_observation_space, n=self.obs_seq_len
        )
        self.observation_space = batch_space(
            self.single_observation_space, n=self.unwrapped.num_envs
        )

        self.single_action_space = batch_space(
            env.single_action_space, n=self.action_horizon or 1
        )
        self.action_space = batch_space(
            self.single_action_space, n=self.unwrapped.num_envs
        )

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options or {})

        # if self.use_render_obs:
        #     rendered_obs = self.env.render()
        #     obs.update(rendered_obs)

        # unsqueeze time dimension (works for both torch and numpy)
        # obs = tree_get_item(obs, (slice(None), None))

        return obs, info

    def step(
        self, actions: Tensor
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Any]]:

        # prepare to loop over time dimension (works for both torch and numpy)
        # (num_envs, time, ... ) -> (time, num_envs, ...)
        actions = actions.swapaxes(0, 1)

        obs, reward, terminated, truncated, info = self.env.step(actions[0])

        dones = torch.logical_or(terminated, truncated)
        not_dones = ~dones

        if not_dones.any():
            # if action_horizon is None, use all actions
            for action in actions[1 : self.action_horizon]:
                obs, new_reward, new_terminations, new_truncations, info = (
                    self.env.step(action)
                )

                reward[not_dones] += new_reward[not_dones]
                terminated = torch.logical_or(terminated, new_terminations)
                truncated = torch.logical_or(truncated, new_truncations)

                dones = torch.logical_or(terminated, truncated)
                not_dones = ~dones

                if dones.all():
                    break

        if dones.any() and self.auto_reset:
            env_idx = dones.nonzero().squeeze(dim=0)

            # reset obs replaces last obs, since it contains the same data
            # again for environments that did not reset
            obs, reset_info = self.env.reset(
                # reset_mask is for Gymnasium's VectorEnv
                # env_idx is for e.g. maniskill ManiSkillVectorEnv
                options={"reset_mask": dones, "env_idx": env_idx}
            )

            assert (
                "episode" not in reset_info
            ), "reset info should not contain episode statistics, since it would overwrite the statistics of completed trajectories in the step info"

            info.update(reset_info)

        # unsqueeze time dimension (works for both torch and numpy)
        # obs = tree_get_item(obs, (slice(None), None))

        # if self.use_render_obs:
        #     rendered_obs = self.env.render()
        #     assert isinstance(obs, MutableMapping)
        #     obs.update(rendered_obs)

        return obs, reward, terminated, truncated, info
