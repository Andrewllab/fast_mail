from utils.point_cloud.sampling.base_pc_sampler import BasePointCloudSampler
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops


class FPSPointCloudSampler(BasePointCloudSampler):
    def _sample(self, point_cloud: np.ndarray, num_points: int) -> np.ndarray:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        point_cloud = torch.from_numpy(point_cloud).unsqueeze(0).to(device)
        num_points = torch.tensor([num_points]).to(device)
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(
            points=point_cloud[..., :3], K=num_points
        )
        point_cloud = point_cloud.squeeze(0).cpu().numpy()
        return point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
