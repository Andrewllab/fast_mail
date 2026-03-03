import logging
from typing import Any, Literal, Mapping

import gymnasium as gym

from environments.wrappers import (
    ActionChunkWrapper,
    RecordEpisodeStatistics,
    RecordMultiEpisodeVideo,
)

log = logging.getLogger(__name__)

OBS_MODES = Literal[
    "state", "state_dict", "none", "sensor_data", "any_textures", "pointcloud"
]
RENDER_MODES = Literal["human", "rgb_array", "sensors", "all"]


def make(
    env_id: str,
    obs_mode: OBS_MODES,
    control_mode: str,
    render_mode: RENDER_MODES | None = None,
    num_envs: int = 1,
    action_horizon: int | None = None,
    obs_seq_len: int = 1,
    max_episode_steps: int | None = None,
    reconfigure_on_reset: bool = False,
    record_video: Mapping[str, Any] | None = None,
    **kwargs,
):
    """
    Creates a ManiSkill environment and applies specified wrappers using Hydra.

    Args:
        env_id: The ID of the ManiSkill environment to create.
        obs_mode: The observation mode (e.g., 'rgbd', 'state').
        control_mode: The control mode for the robot.
        render_mode: The rendering mode.
        num_envs: The number of parallel environments.
        action_horizon: The number of actions to roll out from each chunk.
        obs_seq_len: The number of past observations to include in the observation sequence.
        max_episode_steps: The maximum number of steps per episode.
        reconfigure_on_reset: Whether to reconfigure the environment on reset.
        record_video: A mapping containing video recording configurations.
        **kwargs: Additional keyword arguments for gym.make().
    """
    import mani_skill  # register environments

    from environments.maniskill.wrappers import (
        ManiSkillGoalPosWrapper,
        ManiSkillSpecs,
        ManiSkillVectorEnv,
    )

    log.info(f"Building ManiSkill env '{env_id}'...")

    if reconfigure_on_reset:
        # TODO: apparently maniskill doesn't support resetting individual envs
        # if reconfiguration_freq > 0, so we need to ensure that ActionChunkWrapper
        # (or whatever is doing the resetting) ignores termination signals.
        raise NotImplementedError(
            "reconfigure_on_reset=True is not yet implemented for ManiSkill envs."
        )

    # Creates the base environment using gym.make, which is already vectorized.
    # Timeouts are handled by ManiSkill's version of TimeLimit wrapper.
    envs = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        reconfiguration_freq=(1 if reconfigure_on_reset else None),
        max_episode_steps=max_episode_steps,
        **kwargs,
    )

    # Wrap with ManiSkillVectorEnv so that it conforms to Gymnasium VectorEnv API.
    envs = ManiSkillVectorEnv(
        envs,
        ignore_terminations=False,
    )

    # Replace the goal position given in the observation under "extra/goal_pos"
    # with the one from the environment state dict under "actors/goal_region"
    # (if it exists). In PokeCube-v1, the goal position in the observation is
    # incorrect because it gives the position of the peg and not the target.
    envs = ManiSkillGoalPosWrapper(envs)

    # Record episode statistics such as success and episode length in the info
    # dict under the "episode" key.
    envs = RecordEpisodeStatistics(envs, stats_key="episode")

    record_video = record_video or {}
    record_video = dict(record_video)  # copy to ensure mutability
    if record_video and not record_video.pop("disabled", False):
        envs = RecordMultiEpisodeVideo(envs, **record_video)

    # Handle action chunking and auto-resetting when any env is done.
    envs = ActionChunkWrapper(
        envs,
        action_horizon=action_horizon,
        obs_seq_len=obs_seq_len,
        auto_reset=True,  # reset envs when done
    )

    # Rearrange obs and info to conform to expected format and add DataSpecs
    # derived from observation and action spaces.
    envs = ManiSkillSpecs(envs)

    return envs
