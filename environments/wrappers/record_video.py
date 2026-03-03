from __future__ import annotations

import gc
import logging
import os
import os.path as osp
from typing import TYPE_CHECKING, Any, Literal

import gymnasium as gym
import gymnasium.error as gym_error
import gymnasium.logger as gym_logger
import numpy as np
from gymnasium.vector import VectorWrapper

from utils.rendering import find_tiling, tile_images

if TYPE_CHECKING:
    from gymnasium.vector import ActType, ArrayType, ObsType, VectorEnv


log = logging.getLogger(__name__)


class RecordMultiEpisodeVideo(VectorWrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: VectorEnv[ObsType, ActType, ArrayType],
        video_folder: str,
        num_episodes: int | None = None,
        mode: Literal["tile", "concat", "first"] = "tile",
        video_aspect_ratio: tuple[int, int] = (16, 9),
        fps: int | None = None,
        disable_logger: bool = True,
        gc_trigger: bool = True,
    ):
        # TODO: add resizing of final video to reduce file size

        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_folder=video_folder,
            num_episodes=num_episodes,
            mode=mode,
            video_aspect_ratio=video_aspect_ratio,
            fps=fps,
            disable_logger=disable_logger,
            gc_trigger=gc_trigger,
        )
        VectorWrapper.__init__(self, env)

        if env.render_mode in (None, "human", "ansi"):
            raise ValueError(
                f"Render mode is {env.render_mode}, which is incompatible with RecordVideo.",
                "Initialize your environment with a render_mode that returns an image, such as rgb_array.",
            )

        autoreset_mode = env.metadata.get("autoreset_mode", None)
        if autoreset_mode not in (gym.vector.AutoresetMode.DISABLED, None):
            raise NotImplementedError(
                f"RecordMultiEpisodeVideo does not yet support environments with autoreset_mode={autoreset_mode}."
            )

        if mode not in ("tile", "concat", "first"):
            raise ValueError(
                f"Invalid mode {mode}, expected one of 'tile', 'concat', or 'first'."
            )

        if self.num_envs == 1:
            mode = "first"

        if mode == "tile":
            # find a tiling of the videos of the n environments
            # TODO: implement target aspect ratio tiling
            aspect_ratio: float = video_aspect_ratio[0] / video_aspect_ratio[1]
            self.frame_rows, self.frame_cols = find_tiling(self.num_envs)
            log.info(
                f"Tiling {self.num_envs} videos into {self.frame_rows} rows and {self.frame_cols} columns."
            )
        elif mode == "concat":
            raise NotImplementedError(
                "Mode 'concat' is not yet implemented. Please choose mode 'tile' or 'first'."
            )

        if fps is None:
            fps = self.metadata.get("render_fps", 30)

        self.video_folder = os.path.abspath(video_folder)
        self.num_episodes = num_episodes or float("inf")
        self.mode = mode
        self.frames_per_sec: int = fps
        self.disable_logger = disable_logger
        self.gc_trigger = gc_trigger

        if os.path.isdir(self.video_folder):
            gym_logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.recording: bool = False
        self.recorded_episodes = 0
        self.recorded_frames: list[np.ndarray] = []
        self.render_history: list[np.ndarray] = []

        # These are set by the VideoMetadataWriter callback when evaluation
        # starts.
        self.run_name, self.ckpt_epoch = None, None
        # These are set internally to save a snapshot of the metadata, since
        # the callback may execute between when recording starts and stops.
        self._run_name, self._ckpt_epoch = None, None

        try:
            import moviepy  # noqa: F401
        except ImportError as e:
            raise gym_error.DependencyNotInstalled(
                'MoviePy is not installed, run `pip install "gymnasium[other]"`'
            ) from e

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

    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        envs_frame = self.env.render()

        assert len(envs_frame) == self.num_envs

        if self.mode == "tile":
            render_frame = tile_images(envs_frame, self.frame_rows, self.frame_cols)
        elif self.mode == "concat":
            raise NotImplementedError
        elif self.mode == "first":
            render_frame = envs_frame[0]

        self.recorded_frames.append(render_frame)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and eventually starts a new recording."""
        if options is None or "reset_mask" not in options:
            # reset all environments at the start of evaluation
            self.recorded_episodes = 0

            # always start recording at the beginning of evaluation
            self.start_recording()

        else:
            # auto-reset completed environments during evaluation
            # TODO: adjust this count if only recording the first environment
            self.recorded_episodes += options["reset_mask"].sum().item()
            if self.recording and self.recorded_episodes >= self.num_episodes:
                self.stop_recording()

        obs, info = super().reset(seed=seed, options=options)

        if self.recording:
            self._capture_frame()

        return obs, info

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        obs, rewards, terminations, truncations, info = self.env.step(actions)

        if self.recording:
            self._capture_frame()

        return obs, rewards, terminations, truncations, info

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        if self.recording:
            self.stop_recording()

    def start_recording(self):
        """Start a new recording. If it is already recording, stops the current recording before starting the new one."""
        if self.recording:
            self.stop_recording()

        self.recording = True

        # snapshot the metadata at the start of recording
        self._run_name = self.run_name
        self._ckpt_epoch = self.ckpt_epoch

    def stop_recording(self):
        """Stop current recording and saves the video."""
        assert self.recording, "stop_recording was called, but no recording was started"

        run_name = self._run_name
        ckpt_epoch = self._ckpt_epoch

        if len(self.recorded_frames) <= 1:
            # if metadata has been set but only the reset rendering has been recorded
            if None not in (run_name, ckpt_epoch):
                gym_logger.warn(
                    "Ignored saving a video as there were zero frames to save."
                )

        else:
            # if there are recorded frames but no metadata has been set
            if (run_name, ckpt_epoch) == (None, None):
                run_name = "unknown"
                ckpt_epoch = 0

                log.warning(
                    f"No metadata has been set in the video recorder. Defaulting to run_name={run_name} and ckpt_epoch={ckpt_epoch}"
                )

            video_path = osp.join(
                self.video_folder, f"run_{run_name}_epoch_{ckpt_epoch}.mp4"
            )

            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError as e:
                raise gym_error.DependencyNotInstalled(
                    'MoviePy is not installed, run `pip install "gymnasium[other]"`'
                ) from e

            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            moviepy_logger = None if self.disable_logger else "bar"
            clip.write_videofile(video_path, logger=moviepy_logger)

            log.info(
                f"Saved video recording of {len(self.recorded_frames)} frames to {video_path}."
            )

            if self.rollouts_table is not None:
                import wandb

                video = wandb.Video(video_path, format="mp4")
                self.rollouts_table.add_data(ckpt_epoch, video)

                assert wandb.run is not None
                wandb.run.log({"test/rollouts": self.rollouts_table})

        self.recorded_frames = []
        self.recording = False

        if self.gc_trigger:
            gc.collect()
