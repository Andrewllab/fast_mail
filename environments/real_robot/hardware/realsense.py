import logging
import time
import weakref
from typing import Literal, Mapping, Sequence

import numpy as np
import pyrealsense2 as rs
import torch

from environments.real_robot.hardware.base_camera import BaseCamera
from environments.specs import (
    CameraSpec,
    DepthStream,
    ImageStream,
    PinholeCameraIntrinsic,
    RGBStream,
)

log = logging.getLogger(__name__)


class RealSense(BaseCamera):
    RECORDING_HEIGHT = 480
    RECORDING_WIDTH = 640

    def __init__(
        self,
        serial_number: int | str,
        name: str | None = None,
        # for some reason, the API ignores any other resolution than 480x640
        height: int = RECORDING_HEIGHT,
        width: int = RECORDING_WIDTH,
        fps: int = 30,
        reconnect_attempts: int = 2,
        warm_start: int = 3,
        intrinsics: Mapping[str, float] | None = None,
        extrinsics: Sequence[Sequence[float]] | None = None,
    ):
        self.serial_number = str(serial_number)
        self._name = name if name else f"RealSense_{serial_number}"
        self.fps = fps
        self.reconnect_attempts = reconnect_attempts
        self.warm_start = warm_start

        self._intrinsics = intrinsics
        self._extrinsics = (
            torch.tensor(extrinsics).reshape(4, 4) if extrinsics is not None else None
        )

        self._connect()

    @property
    def height_width(self) -> tuple[int, int]:
        """
        Returns the height and width of the camera image.
        """
        return self.RECORDING_HEIGHT, self.RECORDING_WIDTH

    @property
    def spec(self) -> CameraSpec:
        return self._spec

    def _connect(self):
        """
        Connects to this instance.
        """
        for i in range(self.reconnect_attempts):
            try:
                self._do_connect()
                log.info(f'Connection to RealSense "{self.name}" successful.')
                return
            except Exception:
                if i == self.reconnect_attempts - 1:
                    log.error(
                        f'Failed to connect to RealSense "{self.name}" after {self.reconnect_attempts} attempts.'
                    )
                    raise
                log.exception(
                    f'Attempt {i + 1} to connect to RealSense "{self.name}" (serial no. {self.serial_number}) failed.'
                )
                log.warning(f"Resetting device and retrying...")

            # The D405 can sometimes recover after a hardware reset, which we do via the API.
            time.sleep(0.5)
            for device in rs.context().query_devices():
                if device.get_info(rs.camera_info.serial_number) == self.serial_number:
                    device.hardware_reset()
                    log.info(
                        f'Realsense "{self.name}" (serial no. {self.serial_number}) reset.'
                    )
                    break
            time.sleep(0.5)

    def _do_connect(self):
        self.pipe = rs.pipeline()
        self._finalizer = weakref.finalize(self, self.pipe.stop)

        config = rs.config()

        config.enable_device(self.serial_number)
        config.enable_stream(
            rs.stream.depth,
            self.RECORDING_WIDTH,
            self.RECORDING_HEIGHT,
            rs.format.z16,
            self.fps,
        )
        config.enable_stream(
            rs.stream.color,
            self.RECORDING_WIDTH,
            self.RECORDING_HEIGHT,
            rs.format.rgb8,
            self.fps,
        )
        config.enable_stream(
            rs.stream.infrared,
            1,
            self.RECORDING_WIDTH,
            self.RECORDING_HEIGHT,
            rs.format.y8,
            self.fps,
        )
        config.enable_stream(
            rs.stream.infrared,
            2,
            self.RECORDING_WIDTH,
            self.RECORDING_HEIGHT,
            rs.format.y8,
            self.fps,
        )
        self.profile = self.pipe.start(config)
        self.align = rs.align(rs.stream.color)

        # double check the connection by actually getting some frames
        for _ in range(self.warm_start):
            self._get_frameset()

        if self._intrinsics is not None:
            intrinsics = self._intrinsics
        else:
            # if intrinsics are not provided, get them from the camera
            intrinsics = self.get_intrinsics()

        assert intrinsics["height"] == self.RECORDING_HEIGHT
        assert intrinsics["width"] == self.RECORDING_WIDTH

        self._spec = CameraSpec(
            streams={
                "rgb": RGBStream(
                    self.RECORDING_HEIGHT, self.RECORDING_WIDTH, channels=3
                ),
                "depth": DepthStream(self.RECORDING_HEIGHT, self.RECORDING_WIDTH),
                "left": ImageStream(
                    self.RECORDING_HEIGHT, self.RECORDING_WIDTH, channels=None
                ),
                "right": ImageStream(
                    self.RECORDING_HEIGHT, self.RECORDING_WIDTH, channels=None
                ),
            },
            intrinsics=PinholeCameraIntrinsic(
                height=self.RECORDING_HEIGHT,
                width=self.RECORDING_WIDTH,
                fx=intrinsics["fx"],
                fy=intrinsics["fy"],
                cx=intrinsics["cx"],
                cy=intrinsics["cy"],
            ),
            extrinsics=self._extrinsics,
            baseline=intrinsics["baseline"],
        )

    def get_intrinsics(
        self,
        stream: Literal["Color", "Depth", "Infrared Left", "Infrared Right"] = "Depth",
    ) -> dict[str, int | float | np.ndarray | str]:
        # https://github.com/IntelRealSense/librealsense/issues/12090#issuecomment-1673844543

        STREAM_NAMES = {
            "Color": "Color",
            "Depth": "Depth",
            "Infrared Left": "Infrared 1",
            "Infrared Right": "Infrared 2",
        }

        stream_name = STREAM_NAMES[stream]
        for s in self.profile.get_streams():
            if s.stream_name() == stream_name:
                break
        else:
            raise ValueError(
                f"Stream {stream_name} not found in RealSense profile for {self.name}."
            )

        intrinsics = s.as_video_stream_profile().get_intrinsics()
        intrinsics = {
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.ppx,
            "cy": intrinsics.ppy,
            "width": intrinsics.width,
            "height": intrinsics.height,
            "distortion_model": intrinsics.model.name,
            "distortion_coeffs": np.array(intrinsics.coeffs, dtype=np.float32),
        }

        # compute stereo baseline
        for left_ir in self.profile.get_streams():
            if left_ir.stream_name() == "Infrared 1":
                break
        for right_ir in self.profile.get_streams():
            if right_ir.stream_name() == "Infrared 2":
                break

        # baseline is the x component of the extrinsic transform between the two IR cameras
        baseline = right_ir.get_extrinsics_to(left_ir).translation[0]
        assert baseline > 0
        intrinsics["baseline"] = float(baseline)

        return intrinsics

    def _get_frameset(self):
        try:
            frameset = self.pipe.wait_for_frames()
        except Exception as e:
            log.exception(
                f'Failed to get frameset from RealSense "{self.name}" (serial no. {self.serial_number})'
            )
            raise
        return self.align.process(frameset)

    def get_rgb(self) -> np.ndarray:
        """
        Returns the RGB image as a numpy array.
        """
        frameset = self._get_frameset()
        rgb_frame = frameset.get_color_frame()
        return np.asanyarray(rgb_frame.get_data())

    def get_depth(self) -> np.ndarray:
        """
        Returns the depth image as a numpy array in millimeters.
        """
        frameset = self._get_frameset()
        depth_frame = frameset.get_depth_frame()
        depth = np.asanyarray(depth_frame.get_data(), dtype=np.float32)
        depth *= depth_frame.get_units()
        return depth

    def get_observation(self) -> dict[str, float | np.ndarray]:
        """
        returns color image as np.ndarray [h, w, 3] with RGB[0-255] and depth as np.ndarray [h, w] in millimeters
        """
        frameset = self._get_frameset()

        rgb_frame = frameset.get_color_frame()
        rgb = np.asanyarray(rgb_frame.get_data())

        depth_frame = frameset.get_depth_frame()
        depth = np.asanyarray(depth_frame.get_data(), dtype=np.float32)
        depth *= depth_frame.get_units()

        left_infrared_frame = frameset.get_infrared_frame(1)
        left_infrared = np.asanyarray(left_infrared_frame.get_data())
        right_infrared_frame = frameset.get_infrared_frame(2)
        right_infrared = np.asanyarray(right_infrared_frame.get_data())

        return {
            "time": time.time(),
            "rgb": rgb,
            "depth": depth,
            "left": left_infrared,
            "right": right_infrared,
        }

    def close(self):
        self.pipe.stop()
        self._finalizer.detach()
        del self.pipe
        del self._finalizer

    @staticmethod
    def get_all_devices(count: int | None = None, **kwargs) -> list["RealSense"]:
        """
        Finds and returns specific number of instances of this class.

        Parameters:
        ----------
        - `count` (int): Maximum number of instances to be found. Leaving out `count` or `count = None` returns all instances.
        - `**kwargs`: Keyword arguments passed to the constructor of `RealSense`.

        Returns:
        --------
        - `devices` (list[RealSense]): List of found devices. If no devices are found, `[]` is returned.
        """
        devices = rs.context().query_devices()
        cameras = [
            RealSense(
                device.get_info(rs.camera_info.serial_number),
                **kwargs,
            )
            for device in devices[:count]
        ]
        return cameras

    @staticmethod
    def get_device_by_serial_no(serial_number: int | str, **kwargs) -> "RealSense":
        serial_no = str(serial_number)
        devices = rs.context().query_devices()
        for device in devices:
            if device.get_info(rs.camera_info.serial_number) == serial_no:
                return RealSense(serial_no, **kwargs)
        raise ValueError(f"RealSense with serial number {serial_no} not found.")
