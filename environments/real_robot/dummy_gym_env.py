import logging
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import torch
from moviepy import VideoFileClip

from environments.specs import ActionSpec, CameraSpec, DataSpecs

log = logging.getLogger(__name__)


class DummyGymEnv(gym.Env):
    def __init__(
        self,
        action_seq_len: int,
        video_filepath: os.PathLike,
        fps: float = 30,
        trim_to: float | None = None,
        loop: bool = False,
    ):
        self.video_filepath = Path(video_filepath)
        self.fps = fps
        self.loop = loop

        self.obs_key = self.video_filepath.stem
        self.clip = VideoFileClip(video_filepath)

        if trim_to is not None:
            self.clip = self.clip.subclipped(0, trim_to)

        width, height = self.clip.size
        self.frames_it = self.clip.iter_frames(fps=self.fps, dtype="uint8")

        # limit fps of program loop
        self.clock = pygame.time.Clock()

        self._specs = DataSpecs(
            obs={self.obs_key: CameraSpec(shape=(1, height, width, 3))},
            action=ActionSpec(shape=(action_seq_len, 1), type="action"),
        )

        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(action_seq_len, 1), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
                self.obs_key: gym.spaces.Box(
                    low=0, high=255, shape=(1, height, width, 3), dtype=np.float32
                )
            }
        )

    def get_obs(self) -> dict:
        self.clock.tick(self.fps)
        try:
            frame = next(self.frames_it)
        except StopIteration:
            # end of video
            if self.loop:
                log.debug("Reached end of video, looping")
                self.frames_it = self.clip.iter_frames(fps=self.fps, dtype="uint8")
                frame = next(self.frames_it)
            else:
                log.info("Reached end of video, interrupting program...")
                raise KeyboardInterrupt

        # copy the image because the original is not writable
        image = torch.tensor(frame).unsqueeze(0)
        return {self.obs_key: image}

    def reset(self, *, seed=None, options=None):
        return self.get_obs(), {}

    def step(self, action):
        return self.get_obs(), 0.0, False, False, {}

    @property
    def specs(self) -> DataSpecs:
        return self._specs
