import math
import numpy as np
import open3d as o3d
import torch

def quat2mat_torch(quaternions):
    """
    Converts given quaternion(s) to matrix/matrices.

    Args:
        quaternions (torch.Tensor): (x,y,z,w) vec4 float angles, shape (..., 4)

    Returns:
        torch.Tensor: (..., 3, 3) rotation matrices
    """
    inds = torch.tensor([3, 0, 1, 2], device=quaternions.device)
    q = quaternions[..., inds]

    n = torch.sum(q * q, dim=-1, keepdim=True)
    q = q * torch.sqrt(2.0 / n)
    q2 = torch.einsum('...i,...j->...ij', q, q)

    return torch.stack([
        1.0 - q2[..., 2, 2] - q2[..., 3, 3], q2[..., 1, 2] - q2[..., 3, 0], q2[..., 1, 3] + q2[..., 2, 0],
        q2[..., 1, 2] + q2[..., 3, 0], 1.0 - q2[..., 1, 1] - q2[..., 3, 3], q2[..., 2, 3] - q2[..., 1, 0],
        q2[..., 1, 3] - q2[..., 2, 0], q2[..., 2, 3] + q2[..., 1, 0], 1.0 - q2[..., 1, 1] - q2[..., 2, 2]
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)