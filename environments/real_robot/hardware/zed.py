import logging
import time
import weakref
from typing import Literal, Mapping, Sequence

import cv2
import numpy as np
import pyzed.sl as sl
import torch

from environments.real_robot.hardware.base_camera import BaseCamera
from environments.specs import (
    CameraSpec,
    DepthStream,
    PinholeCameraIntrinsic,
    RGBStream,
)

log = logging.getLogger(__name__)


class Zed(BaseCamera):
    def __init__(
        self,
        serial_number: int | str,
        name: str | None = None,
        resolution: Literal["HD720", "HD1080", "HD2K", "VGA"] = "HD720",
        depth_mode: Literal[
            "QUALITY", "ULTRA", "NONE", "NEURAL", "PERFORMANCE"
        ] = "QUALITY",
        fps: Literal[15, 30, 60, 100] = 30,
        reconnect_attempts: int = 2,
        warm_start: int = 0,
        intrinsics: Mapping[str, float] | None = None,
        extrinsics: Sequence[Sequence[float]] | None = None,
    ):
        self.serial_number = str(serial_number)
        self._name = name if name else f"Zed_{serial_number}"
        self.fps = fps
        self.reconnect_attempts = reconnect_attempts
        self.warm_start = warm_start

        if resolution in sl.RESOLUTION:
            self.resolution = sl.RESOLUTION[resolution]
        else:
            log.warning(f"Resolution {resolution} not found. Using default: HD720")
            self.resolution = sl.RESOLUTION.HD720

        if depth_mode in sl.DEPTH_MODE:
            self.depth_mode = sl.DEPTH_MODE[depth_mode]
        else:
            log.warning(f"Depth mode {depth_mode} not found. Using default: QUALITY")
            self.depth_mode = sl.DEPTH_MODE.QUALITY

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
        return self._height_width

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
                log.info(f"Connection to Zed {self.name} successful.")
                return
            except Exception:
                log.exception(
                    f"Attempt {i + 1} to connect to Zed {self.name} (serial no. {self.serial_number}) failed."
                )
                time.sleep(1)  # wait before retrying

    def _do_connect(self):
        self.zed = sl.Camera()
        self._finalizer = weakref.finalize(self, self.zed.close)

        init_params = sl.InitParameters()
        init_params.camera_resolution = self.resolution
        init_params.camera_fps = self.fps
        init_params.depth_mode = self.depth_mode
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            log.error(f"Failed to open ZED camera: {err}")
            raise RuntimeError(f"Failed to open ZED camera: {err}")

        self.image_left = sl.Mat()
        self.image_right = sl.Mat()
        self.depth = sl.Mat()
        self.runtime_parameters = sl.RuntimeParameters()

        for _ in range(self.warm_start):
            self._retrieve_images()

        obs = self.get_observation()
        height, width = obs["left"].shape[:2]
        assert obs["right"].shape[:2] == (height, width)
        assert obs["depth"].shape[:2] == (height, width)
        self._height_width = (height, width)

        if self._intrinsics is not None:
            intrinsics = self._intrinsics
        else:
            # if intrinsics are not provided, get them from the camera
            intrinsics = self.get_intrinsics()

        assert (intrinsics["height"], intrinsics["width"]) == self._height_width

        self._spec = CameraSpec(
            streams={
                "left": RGBStream(height, width, channels=3),
                "right": RGBStream(height, width, channels=3),
                "depth": DepthStream(height, width),
            },
            intrinsics=PinholeCameraIntrinsic(
                height=intrinsics["height"],
                width=intrinsics["width"],
                fx=intrinsics["fx"],
                fy=intrinsics["fy"],
                cx=intrinsics["cx"],
                cy=intrinsics["cy"],
            ),
            extrinsics=self._extrinsics,
        )

    def get_intrinsics(self) -> dict[str, float]:
        intrinsics = (
            self.zed.get_camera_information().camera_configuration.calibration_parameters
        )

        cx = intrinsics.left_cam.cx
        cy = intrinsics.left_cam.cy
        fx = intrinsics.left_cam.fx
        fy = intrinsics.left_cam.fy
        width = intrinsics.left_cam.image_size.width
        height = intrinsics.left_cam.image_size.height
        distortion = intrinsics.left_cam.disto
        baseline = intrinsics.stereo_transform.get_translation().get()[0] / 1000

        return {
            "cx": cx,
            "cy": cy,
            "fx": fx,
            "fy": fy,
            "width": width,
            "height": height,
            "distortion": distortion,
            "baseline": baseline,
        }

    def _retrieve_images(self):
        if self.zed is None:
            raise RuntimeError(f"Not connected to {self.name}")

        if (err := self.zed.grab(self.runtime_parameters)) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            # self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        else:
            log.error(f"Failed to retrieve images from {self.name}: {err}")
            raise RuntimeError(f"Failed to retrieve images from {self.name}: {err}")

    def get_observation(self):
        """
        Prompts the device to output a single frame of the sensor data.
        Output has the following format: `{'time': timestamp, 'left': rgb_vals, 'right': rgb_vals, 'depth': depth_vals}`
        The depth values are in millimeters.
        The RGB values are in BGRA format.

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format `{'time': float, 'left': uint8, 'right': uint8, 'depth': Any }`.
        """
        self._retrieve_images()

        image_left_np = self.image_left.get_data()  # BGRA
        image_left_np = cv2.cvtColor(image_left_np, cv2.COLOR_BGRA2RGB)  # RGB
        image_right_np = self.image_right.get_data()  # BGRA
        image_right_np = cv2.cvtColor(image_right_np, cv2.COLOR_BGRA2RGB)  # RGB
        depth_np = self.depth.get_data()
        # point_cloud_np = point_cloud.get_data()

        return {
            "time": time.time(),
            "rgb": image_left_np,
            "left": image_left_np,
            "right": image_right_np,
            "depth": depth_np,
        }  # , "point_cloud": point_cloud_np}

    def close(self):
        self.zed.close()
        self._finalizer.detach()

    @staticmethod
    def get_all_devices(count: int | None = None, **kwargs) -> list["Zed"]:
        """
        Finds and returns specific number of instances of this class.

        Parameters:
        ----------
        - `count` (int): Maximum number of instances to be found. Leaving out `count` or `count = None` returns all instances.
        - `**kwargs`: Keyword arguments passed to the constructor of `Zed`.

        Returns:
        --------
        - `devices` (list[Zed]): List of found devices. If no devices are found, `[]` is returned.
        """
        devices = sl.Camera.get_device_list()
        cameras = [
            Zed(
                device.serial_number,
                **kwargs,
            )
            for device in devices[:count]
        ]
        return cameras

    @staticmethod
    def get_device_by_serial_no(serial_number: int | str, **kwargs) -> "Zed":
        serial_no = str(serial_number)
        devices = sl.Camera.get_device_list()
        for device in devices:
            if device.get_camera_information().serial_number == serial_no:
                return Zed(serial_no, **kwargs)
        raise ValueError(f"Zed with serial number {serial_number} not found.")
