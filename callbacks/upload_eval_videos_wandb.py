from lightning.pytorch.callbacks import Callback
from pathlib import Path
import logging
from loggers.wandb import upload_eval_videos

log = logging.getLogger(__name__)


class UploadEvalVideosToWandbOnTestEnd(Callback):
    def __init__(
        self, artifact_run_name: str, artifact_version: str, recording_dir: str
    ):
        self.artifact_run_name = artifact_run_name
        self.artifact_version = artifact_version
        self.recording_dir = Path(recording_dir) / artifact_run_name

    def on_test_end(self, trainer, pl_module):

        log.info("Starting uploading evaluation videos to wandb.")
        upload_eval_videos(self.artifact_version, self.recording_dir)
        log.info("Uploading evaluation videos to wandb completed.")


class LogEvalTableOnPredictEnd(Callback):
    def __init__(
        self, artifact_run_name: str, artifact_version: str, recording_dir: str
    ):
        self.artifact_run_name = artifact_run_name
        self.artifact_version = artifact_version
        self.recording_dir = Path(recording_dir)

    def on_predict_end(self, trainer, pl_module):

        log.info("Starting uploading evaluation videos to wandb.")
        upload_eval_videos(self.artifact_version, self.recording_dir)
        log.info("Uploading evaluation videos to wandb completed.")
