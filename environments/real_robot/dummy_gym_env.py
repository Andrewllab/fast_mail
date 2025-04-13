import logging
import os
from pathlib import Path

import gymnasium as gym
import torch
from moviepy import VideoFileClip

from environments.specs import ActionSpec, DataSpecs, RGBCameraSpec, specs_to_spaces

log = logging.getLogger(__name__)


class DummyGymEnv(gym.Env):
    def __init__(
        self,
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
        self.done = False

        self._specs = DataSpecs(
            obs={self.obs_key: RGBCameraSpec(shape=(height, width, 3))},
            action=ActionSpec(shape=(1,), type="action"),
        )

        self.observation_space, self.action_space = specs_to_spaces(self._specs)

    def get_obs(self) -> dict:
        try:
            frame = next(self.frames_it)
        except StopIteration:
            # end of video
            if self.loop:
                log.debug("Reached end of video, looping")
                self.frames_it = self.clip.iter_frames(fps=self.fps, dtype="uint8")
                frame = next(self.frames_it)
                self.done = True  # the next step after this should return done
            else:
                log.info("Reached end of video, interrupting program...")
                raise KeyboardInterrupt

        # copy the image because the original is not writable
        image = torch.tensor(frame)
        return {self.obs_key: image}

    def reset(self, *, seed=None, options=None):
        return self.get_obs(), {}

    def step(self, action):
        done, self.done = self.done, False
        return self.get_obs(), 0.0, done, False, {}

    @property
    def specs(self) -> DataSpecs:
        return self._specs
