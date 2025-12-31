from __future__ import annotations

import logging
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers import RecordVideo
from torch import Tensor

from environments.specs import CameraSpec, RGBStream

log = logging.getLogger(__name__)


class RecordRobotEnvVideo(RecordVideo):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        video_folder: str,
        # episode_trigger: Callable[[int], bool] | None = None,
        name_prefix: str = "robot-rollout",
        fps: int | None = None,
        disable_logger: bool = True,
        gc_trigger: Callable[[int], bool] | None = lambda episode: True,
    ):
        # TODO: pass random directory as video_folder to suppress warning
        super().__init__(
            env,
            video_folder=video_folder,
            # record every episode
            episode_trigger=lambda _: True,
            step_trigger=None,  # we only use an episode trigger
            # this sets the video_length to infinity so that the parent class
            # never stops recording in the step method
            video_length=0,
            name_prefix=name_prefix,
            fps=fps,
            disable_logger=disable_logger,
            gc_trigger=gc_trigger,
        )

        specs = env.unwrapped.specs
        inputs = {}
        for key, spec in specs.obs.items():
            if not isinstance(spec, CameraSpec):
                continue

            streams = [
                (name, stream)
                for name, stream in spec.streams.items()
                if isinstance(stream, RGBStream)
            ]

            if not streams:
                raise ValueError(f"No RGBStream found in {key}")

            if len(streams) > 1:
                log.warning(
                    f"Camera spec '{key}' contains multiple RGBStreams. "
                    f"Only the first RGBStream '{streams[0][0]}' will be used"
                )

            name, stream = streams[0]
            inputs[(key, name)] = (spec, stream)

        self.inputs = inputs
        self._recorded_frames = defaultdict(list)

        self.root_video_folder = self.video_folder
        self._run_name = None
        self._ckpt_epoch = None

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

    def _capture_frame(self, obs: dict[str, np.ndarray]):
        for key, name in self.inputs.keys():
            image = obs[key][name]
            self._recorded_frames[key].append(image)

    def reset(self, **kwargs):
        # save every episode as a unique video
        self.stop_recording()

        # skip the parent's reset method entirely
        obs, info = super(RecordVideo, self).reset(**kwargs)

        now = datetime.now()
        self.start_recording(f"{self.name_prefix}-{now.strftime('%Y-%m-%d_%H-%M-%S')}")

        self._capture_frame(obs)

        return obs, info

    def step(
        self, action: Tensor
    ) -> tuple[dict[str, np.ndarray], SupportsFloat, bool, bool, dict[str, Any]]:
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        obs, rew, terminated, truncated, info = self.env.step(action)

        # start and stop recording only happen on reset

        # always capture frame because always recording
        self._capture_frame(obs)

        return obs, rew, terminated, truncated, info

    def start_recording(self, video_name: str):
        super().start_recording(video_name)

        self.video_folder = osp.join(
            self.root_video_folder,
            f"{self.run_name}_epoch{self.ckpt_epoch}",
        )
        os.makedirs(self.video_folder, exist_ok=True)

    def stop_recording(self):

        base_video_name = self._video_name

        # call the parent method in a loop so it saves the frames from each camera for us
        for key, name in self.inputs.keys():

            self.recorded_frames = self._recorded_frames[key]

            # set unique filename for this camera
            self._video_name = f"{base_video_name}_{key}_{name}"

            self.recording = True  # parent class asserts self.recording == True
            super().stop_recording()

            # clear stored frames
            self._recorded_frames[key] = []
