import math
import numpy as np
import open3d as o3d
import torch


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

def quat2MatBatch(quats):
    """
    Vectorized version of quat2Mat for a batch of quaternions.

    Input:
        quats: A NumPy array of shape (B, T, 4) where
               B = batch size,
               T = number of time steps (or any other dimension),
               and the last dimension is (w, x, y, z).
    Output:
        A NumPy array of shape (B, T, 3, 3) representing the rotation matrices.
    """

    # Extract w, x, y, z. Each will have shape (B, T).
    w = quats[..., 0]
    x = quats[..., 1]
    y = quats[..., 2]
    z = quats[..., 3]

    # Pre-compute squares
    x2 = x * x
    y2 = y * y
    z2 = z * z
    w2 = w * w

    # Cross terms
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    # Each element of the rotation matrix, shaped (B, T)
    m00 = x2 - y2 - z2 + w2
    m01 = 2.0 * (xy - zw)
    m02 = 2.0 * (xz + yw)

    m10 = 2.0 * (xy + zw)
    m11 = -x2 + y2 - z2 + w2
    m12 = 2.0 * (yz - xw)

    m20 = 2.0 * (xz - yw)
    m21 = 2.0 * (yz + xw)
    m22 = -x2 - y2 + z2 + w2

    # Stack them along the last dimension and reshape to (B, T, 3, 3)
    # shape before reshape: (B, T, 9)
    rot_mat = torch.stack([m00, m01, m02,
                        m10, m11, m12,
                        m20, m21, m22], axis=-1)
    rot_mat = rot_mat.reshape(quats.shape[:-1] + (3, 3))
    return rot_mat


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
    return posRotMat2Mat(pos, rot)