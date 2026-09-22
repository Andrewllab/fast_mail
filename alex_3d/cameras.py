"""RGB-D camera interfaces and an Intel RealSense D405 implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from alex_3d.track_online_pipe import CameraCalibration


@dataclass(frozen=True)
class RGBDFrame:
    """One synchronized RGB-D capture.

    Depth is optical-axis Z in metres and is aligned to the RGB image.
    ``camera_to_world`` maps OpenCV camera coordinates (right, down, forward)
    into the chosen world frame.
    """

    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    index: int
    timestamp: float
    hardware_timestamp_ms: float | None = None


class BaseRGBDCamera(ABC):
    """Interface shared by real and future simulated RGB-D cameras."""

    def __init__(self, name: str):
        if not name:
            raise ValueError("camera name must be nonempty")
        self.name = name
        self.initialized = False

    @abstractmethod
    def initialize(self) -> CameraCalibration:
        """Start acquisition and return calibration in processed RGB pixels."""

    @abstractmethod
    def read(self) -> RGBDFrame:
        """Return the next synchronized RGB-D frame."""

    @abstractmethod
    def close(self) -> None:
        """Stop acquisition and release hardware resources."""

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class RealSenseD405Camera(BaseRGBDCamera):
    """Aligned RGB-D acquisition from an Intel RealSense D405.

    ``pyrealsense2`` is imported only by :meth:`initialize`, so the rest of the
    package and its tests work on machines without RealSense drivers.
    """

    def __init__(
        self,
        name: str = "d405",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: str | None = None,
        camera_to_world: np.ndarray | None = None,
        warmup_frames: int = 15,
        timeout_ms: int = 5000,
    ):
        super().__init__(name)
        if width < 2 or height < 2 or fps <= 0 or warmup_frames < 0 or timeout_ms <= 0:
            raise ValueError("invalid RealSense resolution, fps, warmup, or timeout")
        self.width, self.height, self.fps = width, height, fps
        self.serial = serial
        self.camera_to_world = np.asarray(
            np.eye(4) if camera_to_world is None else camera_to_world,
            dtype=np.float64,
        )
        if self.camera_to_world.shape != (4, 4):
            raise ValueError("camera_to_world must be [4,4]")
        self.warmup_frames = warmup_frames
        self.timeout_ms = timeout_ms
        self._pipeline: Any = None
        self._align: Any = None
        self._depth_scale: float | None = None
        self._intrinsics: np.ndarray | None = None
        self._index = -1

    def initialize(self) -> CameraCalibration:
        if self.initialized:
            return self.calibration
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError(
                "RealSense support requires pyrealsense2; install requirements_robot.txt"
            ) from error

        pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = pipeline.start(config)
        self._pipeline = pipeline
        try:
            device_name = profile.get_device().get_info(rs.camera_info.name)
            if "D405" not in device_name.upper():
                raise RuntimeError(f"Expected a RealSense D405, found {device_name!r}")
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())
            self._align = rs.align(rs.stream.color)
            for _ in range(self.warmup_frames):
                self._pipeline.wait_for_frames(self.timeout_ms)
            frames = self._align.process(self._pipeline.wait_for_frames(self.timeout_ms))
            color = frames.get_color_frame()
            if not color:
                raise RuntimeError("D405 did not produce a color frame")
            intr = color.profile.as_video_stream_profile().intrinsics
            self._intrinsics = np.array(
                [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            self.calibration = CameraCalibration(
                self._intrinsics, self.camera_to_world,
                depth_scale=1.0,
                orthographic=True,
            )
            self.initialized = True
            return self.calibration
        except Exception:
            if self._pipeline is not None:
                self._pipeline.stop()
            self._pipeline = None
            raise

    def read(self) -> RGBDFrame:
        if not self.initialized:
            raise RuntimeError("call initialize() before read()")
        frames = self._align.process(self._pipeline.wait_for_frames(self.timeout_ms))
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("D405 returned an incomplete aligned RGB-D frame")
        rgb = np.ascontiguousarray(np.asanyarray(color.get_data()))
        depth_m = np.asanyarray(depth.get_data()).astype(np.float32) * self._depth_scale
        depth_m[~np.isfinite(depth_m) | (depth_m <= 0)] = np.nan
        if rgb.shape[:2] != depth_m.shape:
            raise RuntimeError(f"aligned RGB/depth mismatch: {rgb.shape} vs {depth_m.shape}")
        self._index += 1
        return RGBDFrame(
            rgb=rgb,
            depth=np.ascontiguousarray(depth_m),
            intrinsics=self._intrinsics.copy(),
            camera_to_world=self.camera_to_world.copy(),
            index=self._index,
            timestamp=time.time(),
            hardware_timestamp_ms=float(color.get_timestamp()),
        )

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
        self._pipeline = None
        self._align = None
        self.initialized = False


def camera_from_config(config: dict) -> BaseRGBDCamera:
    camera_type = config.get("type", "realsense_d405")
    if camera_type != "realsense_d405":
        raise ValueError(f"unsupported camera type {camera_type!r}")
    extrinsics = config.get("camera_to_world")
    return RealSenseD405Camera(
        name=config.get("name", "d405"),
        width=int(config.get("width", 640)),
        height=int(config.get("height", 480)),
        fps=int(config.get("fps", 30)),
        serial=config.get("serial"),
        camera_to_world=None if extrinsics is None else np.asarray(extrinsics),
        warmup_frames=int(config.get("warmup_frames", 15)),
        timeout_ms=int(config.get("timeout_ms", 5000)),
    )
