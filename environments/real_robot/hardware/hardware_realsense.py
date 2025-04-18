# Original Author: Marcel Ruehle
import logging
import time
from typing import Sequence

import numpy as np
import pyrealsense2 as rs
import torch

from environments.real_robot.hardware.hardware_cameras import DiscreteCamera
from environments.specs import (
    CameraSpec,
    DepthStream,
    ImageStream,
    PinholeCameraIntrinsic,
    RGBStream,
)

log = logging.getLogger(__name__)


class RealSense(DiscreteCamera):
    """
    Wrapper that implements boilerplate code for RealSense cameras.
    The recording fails, when using different dimensions than 480x640, so these are now hardcoded to capture the frames.

    Warning: This script is a bit buggy. Sometimes, this error appears (sometimes it doesn't):

    ``File "/home/kkuryshev/audio-pipeline/real_robot/real_robot_env/robot/hardware_realsense.py", line 48, in __get_frames`` \n
      ``return self.align.process(frameset)``
    ``RuntimeError: Error occured during execution of the processing block! See the log for more info``
    """

    RECORDING_HEIGHT = 480
    RECORDING_WIDTH = 640

    def __init__(
        self,
        serial_number: int | str,
        name: str | None = None,
        height=RECORDING_HEIGHT,
        width=RECORDING_WIDTH,
        fps=30,
        extrinsics: Sequence[Sequence[float]] | None = None,
        warm_start=30,
        start_frame_latency=0,
    ):
        super().__init__(
            device_id=str(serial_number),
            name=name if name else f"RealSense_{serial_number}",
            height=height,
            width=width,
            start_frame_latency=start_frame_latency,
        )
        self.fps = fps
        self._extrinsics = (
            torch.tensor(extrinsics).reshape(4, 4) if extrinsics is not None else None
        )
        self.warm_start = warm_start
        self.pipe = None

        self._spec = None

    def connect(self) -> bool:
        """
        Connects to this instance.

        Returns:
        --------
        - `success` (bool): Indicates a successful connection.
        """
        log.info(f"Connecting to RealSense {self.name}...")
        try:
            self._setup_connect()
            log.info(f"Connection to RealSense {self.name} successful.")
            return True

        except Exception as e:
            log.exception(f"Connection to RealSense {self.name} failed.")

        log.info(f"Resetting RealSense {self.name}...")
        devices = rs.context().query_devices()
        for device in devices:
            if device.get_info(rs.camera_info.serial_number) == self.device_id:
                device.hardware_reset()
        log.info(f"Retrying connection to RealSense {self.name}...")
        self._setup_connect()
        log.info(f"Connection to RealSense {self.name} successful.")
        return True

    def _setup_connect(self):
        self.pipe = rs.pipeline()
        config = rs.config()

        config.enable_device(self.device_id)
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

        for _ in range(self.warm_start):
            self.__get_frames()

        intrinsics = self.get_intrinsics_dict()

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
                cx=intrinsics["ppx"],
                cy=intrinsics["ppy"],
            ),
            extrinsics=self._extrinsics,
        )

    def get_intrinsics_dict(self):
        # https://github.com/IntelRealSense/librealsense/issues/12090#issuecomment-1673844543
        # frameset = self.__get_frames()
        # profile = frameset.get_profile()
        stream = self.profile.get_streams()[0]
        intrinsics = stream.as_video_stream_profile().get_intrinsics()
        param_dict = dict(
            [
                (p, getattr(intrinsics, p))
                for p in dir(intrinsics)
                if not p.startswith("__")
            ]
        )
        param_dict["model"] = param_dict["model"].name
        return param_dict

    @property
    def spec(self) -> CameraSpec:
        assert self._spec is not None
        return self._spec

    def __get_frames(self):
        if self.pipe is None:
            raise RuntimeError("Please connect first.")
        frameset = self.pipe.wait_for_frames()
        return self.align.process(frameset)

    def get_obs(self):
        """
        returns color image as np.ndarray [h, w, 3] with RGB[0-255] and depth as np.ndarray [h, w] in millimeters
        """
        frameset = self.__get_frames()

        color_frame = frameset.get_color_frame()
        rgb = np.asanyarray(color_frame.get_data())

        depth_frame = frameset.get_depth_frame()
        d = np.asanyarray(depth_frame.get_data()) * depth_frame.get_units()

        IR1_frame = frameset.get_infrared_frame(1)
        infrared_1 = np.asanyarray(IR1_frame.get_data())
        IR2_frame = frameset.get_infrared_frame(2)
        infrared_2 = np.asanyarray(IR2_frame.get_data())

        return rgb, d, infrared_1, infrared_2

    def get_sensors(self):
        """
        Prompts the device to output a single frame of the sensor data.
        Output has the following format: `{'time': timestamp, 'rgb': rgb_vals, 'd': depth_vals}`

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format `{'time': float, 'rgb': NDArray[uint16], 'd': NDArray[uint16]}`.
        """
        # get all data from all topics
        rgb, d, ir1, ir2 = self.get_obs()

        timestamp = time.time()
        return {
            "time": timestamp,
            "rgb": rgb,
            "depth": d.astype(np.float32),
            "left": ir1,
            "right": ir2,
        }

    def close(self):
        if self.pipe is not None:
            self.pipe.stop()

    @staticmethod
    def get_devices(
        amount=-1, height: int = 480, width: int = 640, **kwargs
    ) -> list["RealSense"]:
        """
        Finds and returns specific amount of instances of this class.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        - `height` (int): Pixel-height of captured frames. Default: `480`
        - `width` (int): Pixel-width of captured frames. Default: `640`
        - `**kwargs`: Arbitrary keyword arguments.

        Returns:
        --------
        - `devices` (list[RealSense]): List of found devices. If no devices are found, `[]` is returned.
        """
        super(RealSense, RealSense).get_devices(amount, height, width, type="RealSense")

        devices = rs.context().query_devices()
        amount = amount if amount != -1 else len(devices)
        cameras = [
            RealSense(
                device.get_info(rs.camera_info.serial_number),
                height=height,
                width=width,
                **kwargs,
            )
            for device in devices[:amount]
        ]
        return cameras

    @staticmethod
    def get_device(serial_number: int, **kwargs) -> "RealSense":
        devices = rs.context().query_devices()
        for device in devices:
            if device.get_info(rs.camera_info.serial_number) == str(serial_number):
                return RealSense(
                    device.get_info(rs.camera_info.serial_number),
                    **kwargs,
                )
        raise ValueError(f"RealSense with serial number {serial_number} not found.")


if __name__ == "__main__":

    pipe = rs.pipeline()
    profile = pipe.start()
    try:
        for i in range(0, 100):
            frames = pipe.wait_for_frames()
            for f in frames:
                print(f.profile)
    finally:
        pipe.stop()

    # cam = RealSense(device_id=0)
    # cam.connect()
    # print(cam.get_intrinsics_dict())

    # for i in range(100):
    #     rgb, d = cam.get_rgbd()
    #     print(i, rgb.shape)
