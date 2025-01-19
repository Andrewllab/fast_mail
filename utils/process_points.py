import torch
import torch.nn as nn
import pytorch3d.ops as torch3d_ops


def Group(xyz, num_group):

    xyz = torch.from_numpy(xyz).float()

    batch_size, num_points, _ = xyz.shape
    # fps the centers out
    center, center_inds = torch3d_ops.sample_farthest_points(points=xyz, K=num_group)  # B G 3

    # knn to get the neighborhood
    idx = torch3d_ops.knn_points(xyz, center)  # B G M

    assert idx.size(1) == num_group
    assert idx.size(2) == group_size
    idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
    neighborhood = neighborhood.view(batch_size, num_group, group_size, 3).contiguous()
    # normalize
    neighborhood = neighborhood - center.unsqueeze(2)
    return neighborhood, center
    

