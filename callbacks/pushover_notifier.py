import os
import logging
import requests
from typing import Optional

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks.callback import Callback
from typing_extensions import override

log = logging.getLogger(__name__)


class PushoverNotifier(Callback):
    def __init__(self, ttl_seconds: Optional[int] = 86400) -> None:
        """
        Initializes the PushoverNotifier callback.

        Args:
            ttl_seconds: Optional. The Time-to-Live for notifications in seconds.
        """
        super().__init__()
        self.pushover_token = os.environ.get("PUSHOVER_TOKEN")
        self.pushover_user = os.environ.get("PUSHOVER_USER")
        self.ttl_seconds = ttl_seconds

        if not self.pushover_token or not self.pushover_user:
            raise ValueError(
                "Pushover token and user key must be set as environment variables "
                "(PUSHOVER_TOKEN, PUSHOVER_USER)."
            )
        log.info(
            f"PushoverNotifier initialized with credentials from environment variables. "
            f"TTL set to {self.ttl_seconds} seconds."
            if self.ttl_seconds
            else "TTL not set."
        )

    # if you want to send more info: https://pushover.net/api
    def _send_pushover_notification(
        self, message: str, title: str = "PyTorch Lightning Notification"
    ) -> None:
        """Helper to send a Pushover notification."""
        url = "https://api.pushover.net/1/messages.json"
        data = {
            "token": self.pushover_token,
            "user": self.pushover_user,
            "message": message,
            "title": title,
        }
        if self.ttl_seconds is not None:
            data["ttl"] = self.ttl_seconds

        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
            log.info(f"Pushover notification sent: {message}")
            log.debug(f"Pushover response: {response.json()}")
        except requests.exceptions.RequestException as e:
            log.error(f"Failed to send Pushover notification: {e}")

    @override
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._send_pushover_notification(
            f"Training started for model '{pl_module.__class__.__name__}'.",
            title="Training Initiated",
        )

    @override
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when fit ends (after all training and validation epochs)."""
        log.info("on_fit_end hook triggered for PushoverNotifier.")
        message = f"Training finished for model '{pl_module.__class__.__name__}'!"
        if trainer.state.status == "finished":
            message += "\nStatus: Successfully completed."
        elif trainer.state.status == "interrupted":
            message += "\nStatus: Interrupted by user or exception."
        elif trainer.state.status == "failed":
            message += "\nStatus: Failed with an error."
        else:
            message += f"\nStatus: {trainer.state.status}"

        self._send_pushover_notification(message, title="Training Complete!")

    @override
    def on_predict_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when predict ends."""
        log.info("on_predict_end hook triggered for PushoverNotifier.")
        message = (
            f"Prediction finished for model '{pl_module.__class__.__name__}'!\n"
            f"Status: Successfully completed."
        )
        self._send_pushover_notification(message, title="Prediction Complete!")

    @override
    def on_exception(
        self, trainer: Trainer, pl_module: LightningModule, exception: BaseException
    ) -> None:
        """Called when any trainer execution is interrupted by an exception."""
        log.error(
            "on_exception hook triggered for PushoverNotifier due to a critical error."
        )

        model_name = pl_module.__class__.__name__
        exception_type = type(exception).__name__
        exception_message = str(exception)

        current_epoch = trainer.current_epoch
        global_step = trainer.global_step

        title = f"Training FAILED: {model_name}"
        message = (
            f"An exception occurred during training.\n\n"
            f"Model: {model_name}\n"
            f"Epoch: {current_epoch}, Step: {global_step}\n\n"
            f"Error Type: {exception_type}\n"
            f"Message: {exception_message}"
        )

        self._send_pushover_notification(message, title=title)
