"""Batched-camera variant of :mod:`alex_3d.track_online_pipe`."""

from __future__ import annotations

from typing import Mapping

from alex_3d.track_online_pipe import (
    CameraCalibration,
    OnlineTrackingPipeline as _OnlineTrackingPipeline,
)
from alex_batch.track_online_pipe import (
    OnlineKeypointTracker,
    SAM3Segmenter,
    TrackFrame,
)


class OnlineTrackingPipeline(_OnlineTrackingPipeline):
    """Use one batched CoTracker predictor while lifting each view separately."""

    def __init__(self, segmenter, tracker: OnlineKeypointTracker,
                 calibrations: Mapping[str, CameraCalibration | Mapping],
                 keep_history: bool = True):
        super().__init__(segmenter, tracker, calibrations, keep_history=keep_history)


__all__ = [
    "CameraCalibration", "SAM3Segmenter", "OnlineKeypointTracker",
    "OnlineTrackingPipeline", "TrackFrame",
]
