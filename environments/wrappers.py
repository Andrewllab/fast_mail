import logging
import os
import os.path as osp
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium.core import ActType, ObsType
from gymnasium.logger import warn
from gymnasium.vector import AutoresetMode, VectorEnv, VectorWrapper
from gymnasium.wrappers import RecordVideo
from tensordict import TensorDict

log = logging.getLogger(__name__)


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


class RecordMultiEpisodeVideo(RecordVideo):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        video_folder: str,
        episode_trigger: Callable[[int], bool] | None = None,
        num_episodes: int = 1,
        name_prefix: str = "rl-video",
        fps: int | None = None,
        disable_logger: bool = True,
        gc_trigger: Callable[[int], bool] | None = lambda episode: True,
    ):
        # TODO: pass random directory as video_folder to suppress warning
        super().__init__(
            env,
            video_folder=video_folder,
            episode_trigger=episode_trigger,
            step_trigger=None,  # we only use an episode trigger
            # this sets the video_length to infinity so that the parent class
            # never stops recording in the step method
            video_length=0,
            name_prefix=name_prefix,
            fps=fps,
            disable_logger=disable_logger,
            gc_trigger=gc_trigger,
        )

        self.num_episodes = num_episodes
        self.recorded_episodes = 0

        self.root_video_folder = self.video_folder
        self._run_name = None
        self._ckpt_epoch = None

        self.rollouts_table = None
        try:
            import wandb

            if wandb.run is not None:
                self.rollouts_table = wandb.Table(
                    columns=["epoch", f"{num_episodes} episodes at {fps}fps"],
                    log_mode="INCREMENTAL",
                )

        except ImportError:
            pass

    @property
    def run_name(self) -> str:
        if self._run_name is None:
            log.warning("run_name is not set for the video recorder")
            return "unknown_run"
        return self._run_name

    @run_name.setter
    def run_name(self, value: str):
        self._run_name = value

    @property
    def ckpt_epoch(self) -> int:
        if self._ckpt_epoch is None:
            log.warning("ckpt_epoch is not set for the video recorder")
            return 0
        return self._ckpt_epoch

    @ckpt_epoch.setter
    def ckpt_epoch(self, value: int):
        self._ckpt_epoch = value

    def reset(self, **kwargs):
        # skip the parent's reset method entirely
        obs, info = super(RecordVideo, self).reset(**kwargs)
        self.episode_id += 1
        self.recorded_episodes += 1

        if self.recording and self.recorded_episodes >= self.num_episodes:
            self.stop_recording()

        if (
            not self.recording
            and self.episode_trigger
            and self.episode_trigger(self.episode_id)
        ):
            self.start_recording(f"{self.name_prefix}-{self.episode_id}")

        if self.recording:
            self._capture_frame()

        return obs, info

    def start_recording(self, video_name: str):
        super().start_recording(video_name)
        self.recorded_episodes = 0

    def stop_recording(self):

        self.video_folder = osp.join(
            self.root_video_folder,
            self.run_name,
            f"epoch_{self.ckpt_epoch}",
        )
        os.makedirs(self.video_folder, exist_ok=True)

        # self._video_name gets reset to None by super().stop_recording()
        path = osp.join(self.video_folder, f"{self._video_name}.mp4")

        super().stop_recording()

        if self.rollouts_table is not None:
            import wandb

            video = wandb.Video(path, format="mp4")
            ckpt_epoch = self.ckpt_epoch
            self.rollouts_table.add_data(ckpt_epoch, video)

            assert wandb.run is not None
            wandb.run.log({"test/rollouts": self.rollouts_table})


class EmptyPointCloudChecker(gym.Wrapper):
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
        super().__init__(env)
        self.min_points = min_points
        self.max_depth = max_depth

    def step(self, action: ActType):
        obs, reward, terminated, truncated, info = super().step(action)

        n_points = 0
        for key, value in obs.items():
            if "depth" in key:
                n_points += (
                    np.logical_and(0 < value, value < self.max_depth).sum().item()
                )

        if n_points < self.min_points:
            log.warning(
                f"Fewer than {self.min_points} are within max depth of {self.max_depth} ({n_points} points). Terminating episode."
            )
            terminated = True

        return obs, reward, terminated, truncated, info
