# Original Author: Marcel Ruehle
import time
import pyzed.sl as sl

from environments.real_robot.hardware.hardware_cameras import DiscreteCamera


class Zed(DiscreteCamera):
    """
    This class can be considered a wrapper class for ZED cameras specifically for frame collection.

    This class inherits its functions from `real_robot_env.robot.hardware_cameras.DiscreteCamera`.
    """

    def __init__(
        self,
        device_id,
        name=None,
        resolution = "VGA",
        fps=30,
        start_frame_latency=0,
    ):
        super().__init__(
            device_id,
            name if name else f"ZED_{device_id}",
            resolution,
            fps,
            start_frame_latency,
        )
        if resolution == "VGA":
            self.resolution = sl.RESOLUTION.VGA
        elif resolution == "HD720":
            self.resolution = sl.RESOLUTION.HD720
        elif resolution == "HD1080":
            self.resolution = sl.RESOLUTION.HD1080
        elif resolution == "HD2K":
            self.resolution = sl.RESOLUTION.HD2K

        self.fps = fps
        self.pipe = None

    def _setup_connect(self):
        self.zed = sl.Camera()

        init_params = sl.InitParameters()
        init_params.camera_resolution = self.resolution # Use HD720 opr HD1200 video mode, depending on camera type.
        init_params.camera_fps = self.fps # Set fps at 30
        #init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE # Set the depth mode to performance (fastest)
        init_params.coordinate_units = sl.UNIT.MILLIMETER

        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Error {}, exit program".format(err)) # Display the error
            exit()

        self.image_left = sl.Mat()
        self.image_right = sl.Mat()
        self.depth = sl.Mat()
        self.runtime_parameters = sl.RuntimeParameters()

    def get_intrinsics_dict(self):
        intrinsics = self.zed.get_camera_information().camera_configuration.calibration_parameters

        cx = intrinsics.left_cam.cx 
        cy = intrinsics.left_cam.cy 
        fx = intrinsics.left_cam.fx 
        fy = intrinsics.left_cam.fy 
        distortion = intrinsics.left_cam.disto
        baseline = intrinsics.stereo_transform.get_translation().get()[0]

        return {"cx": cx, "cy": cy, "fx": fx, "fy": fy, "distortion": distortion, "baseline": baseline}


    def get_sensors(self):
        """
        Prompts the device to output a single frame of the sensor data.
        Output has the following format: `{'time': timestamp, 'left': rgb_vals, 'right': rgb_vals, 'depth': depth_vals}`
        The depth values are in millimeters.
        The RGB values are in BGRA format.

        Returns:
        -------
        - `sensor_data` (dict): Sensor data in the format `{'time': float, 'left': uint8, 'right': uint8, 'depth': Any }`.
        """
        if not self.zed:
            raise Exception(f"Not connected to {self.name}")
        
        # image_left = sl.Mat()
        # image_right = sl.Mat()
        # depth = sl.Mat()
        # #point_cloud = sl.Mat()
        # runtime_parameters = sl.RuntimeParameters()

        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
            self.zed.retrieve_image(self.image_right, sl.VIEW.RIGHT)
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            # self.zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            timestamp = time.time()

            image_left_np = self.image_left.get_data() # BGRA
            image_right_np = self.image_right.get_data() # BGRA
            depth_np = self.depth.get_data()
            # point_cloud_np = point_cloud.get_data()

        return {"time": timestamp, "left": image_left_np, "right": image_right_np, "depth": depth_np}# "point_cloud": point_cloud_np}

    def close(self):
        self.zed.close()
        return True

    @staticmethod
    def get_devices(
        amount=-1, resolution = "VGA", **kwargs
    ) -> list["Zed"]:
        """
        Finds and returns specific amount of instances of this class.

        Parameters:
        ----------
        - `amount` (int): Maximum amount of instances to be found. Leaving out `amount` or `amount = -1` returns all instances.
        - `resolution` (String): Resolution of the camera. Can be "VGA", "HD720", "HD1080", or "HD2K".
        - `fps` (int): Frames per second. Default is 30. 15, 30, 60 and 100 are supported.
        - `**kwargs`: Arbitrary keyword arguments.

        Returns:
        --------
        - `devices` (list[RealSense]): List of found devices. If no devices are found, `[]` is returned.
        """
        super(Zed, Zed ).get_devices(
            amount, resolution = "VGA", type="ZED", **kwargs
        )
        cam_list = sl.Camera.get_device_list()

        init_params = sl.InitParameters()

        print(cam_list)

        cams = []
        counter = 0
        for i in range(len(cam_list)):
            if amount != -1 and counter >= amount:
                break
            init_params.set_from_camera_id(i)
            zed = sl.Camera()
            err = zed.open(init_params)
            if err == sl.ERROR_CODE.SUCCESS:
                cam_info = zed.get_camera_information()
            cam = Zed(device_id=cam_info.serial_number, resolution=resolution, **kwargs)
            cams.append(cam)
            zed.close()
            counter += 1
        return cams
    



if __name__ == "__main__":
    print("ZED camera test")