import math

from utils.camera_utils import (
    quat2Mat,
    posRotMat2Mat,
    get_intrinsic,
)
import open3d as o3d
import numpy as np


class PointCloudGenerator:
    def __init__(self, sim, cam_names: list[str], img_width: int, img_height: int):
        self.sim = sim
        self.cam_names = cam_names

        self.intrinsics = {}

        for cam in cam_names:
            cam_id = self.sim.model.camera_name2id(cam)
            fovy = math.radians(self.sim.model.cam_fovy[cam_id])
            self.intrinsics[cam] = get_intrinsic(fovy, img_width, img_height)

    def get_point_cloud(
        self, imgs: dict[str, np.ndarray], depths: dict[str, np.ndarray]
    ):
        o3d_point_cloud = o3d.geometry.PointCloud()
        colors = []

        for cam in self.cam_names:
            colors.append(imgs[cam])

            o3d_depth = o3d.geometry.Image(depths[cam])
            o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
                o3d_depth, self.intrinsics[cam]
            )

            cam_id = self.sim.model.camera_name2id(cam)
            cam_pos = self.sim.data.cam_xpos[cam_id]
            cam_rot_mat = self.sim.data.cam_xmat[cam_id].reshape(3, 3)
            cam_rot_mat = np.matmul(cam_rot_mat, quat2Mat([0, 1, 0, 0]))
            pose = posRotMat2Mat(cam_pos, cam_rot_mat)

            transformed_cloud = o3d_cloud.transform(pose)
            o3d_point_cloud += transformed_cloud

        pc_points = np.asarray(o3d_point_cloud.points)
        pc_colors = np.array(colors).reshape(-1, 3)

        pc = np.concatenate([pc_points, pc_colors], axis=1)
        return pc
