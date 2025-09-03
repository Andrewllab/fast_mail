import logging
import os.path as osp

from gymnasium.wrappers import RecordVideo
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from agents.base_agent import BaseAgent

log = logging.getLogger(__name__)


class VideoMetadataWriter(Callback):
    def __init__(self, video_recorder: RecordVideo):
        self.video_recorder = video_recorder
        self.orig_video_folder = video_recorder.video_folder

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        assert isinstance(pl_module, BaseAgent)
        metadata = pl_module.checkpoint_metadata

        # set metadata for the video recorder to use to customize the video path
        self.video_recorder.run_name = metadata["run_name"]  # type: ignore
        self.video_recorder.ckpt_epoch = metadata["epoch"]  # type: ignore

        # reset episode counter so that start trigger fires again
        self.video_recorder.episode_id = -1
