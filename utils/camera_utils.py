import math
import numpy as np
import open3d as o3d


def rotMatList2NPRotMat(rot_mat_arr):
    """
    Generates numpy rotation matrix from rotation matrix as list len(9)

    @param rot_mat_arr: rotation matrix in list len(9) (row 0, row 1, row 2)

    @return np_rot_mat: 3x3 rotation matrix as numpy array
    """
    np_rot_arr = np.array(rot_mat_arr)
    np_rot_mat = np_rot_arr.reshape((3, 3))
    return np_rot_mat


def quat2Mat(quat):
    """
    Generates numpy rotation matrix from quaternion

    @param quat: w-x-y-z quaternion rotation tuple

    @return np_rot_mat: 3x3 rotation matrix as numpy array
    """
    if len(quat) != 4:
        print("Quaternion", quat, "invalid when generating transformation matrix.")
        raise ValueError

    # Note that the following code snippet can be used to generate the 3x3
    #    rotation matrix, we don't use it because this file should not depend
    #    on mujoco.
    """
    from mujoco_py import functions
    res = np.zeros(9)
    functions.mju_quat2Mat(res, camera_quat)
    res = res.reshape(3,3)
    """

    # This function is lifted directly from scipy source code
    # https://github.com/scipy/scipy/blob/v1.3.0/scipy/spatial/transform/rotation.py#L956
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    xy = x * y
    zw = z * w
    xz = x * z
    yw = y * w
    yz = y * z
    xw = x * w

    rot_mat_arr = [
        x2 - y2 - z2 + w2,
        2 * (xy - zw),
        2 * (xz + yw),
        2 * (xy + zw),
        -x2 + y2 - z2 + w2,
        2 * (yz - xw),
        2 * (xz - yw),
        2 * (yz + xw),
        -x2 - y2 + z2 + w2,
    ]
    np_rot_mat = rotMatList2NPRotMat(rot_mat_arr)
    return np_rot_mat


def cammat2o3d(cam_mat, width, height):
    """
    Generates Open3D camera intrinsic matrix object from numpy camera intrinsic
        matrix and image width and height

    @param cam_mat: 3x3 numpy array representing camera intrinsic matrix
    @param width:   image width in pixels
    @param height:  image height in pixels

    @return t_mat:  4x4 transformation matrix as numpy array
    """
    cx = cam_mat[0, 2]
    fx = cam_mat[0, 0]
    cy = cam_mat[1, 2]
    fy = cam_mat[1, 1]

    return o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)


def posRotMat2Mat(pos, rot_mat):
    """
    Generates numpy transformation matrix from position list len(3) and
        numpy rotation matrix

    @param pos:     list len(3) containing position
    @param rot_mat: 3x3 rotation matrix as numpy array

    @return t_mat:  4x4 transformation matrix as numpy array
    """
    t_mat = np.eye(4)
    t_mat[:3, :3] = rot_mat
    t_mat[:3, 3] = np.array(pos)
    return t_mat

def get_intrinsic(fovy, img_width, img_height):
    f = img_height / (2 * math.tan(fovy / 2))
    cam_mat = np.array(
        ((f, 0, img_width / 2), (0, f, img_height / 2), (0, 0, 1))
    )
    return cammat2o3d(cam_mat, img_width, img_height)

def get_pose(pos, quat):
    rot = quat2Mat(quat)
    rot = np.matmul(rot, quat2Mat([0, 1, 0, 0]))
    return posRotMat2Mat(pos, quat)