from utils.point_cloud.sampling.base_pc_sampler import BasePointCloudSampler
import numpy as np


class UniformPointCloudSampler(BasePointCloudSampler):
    def _sample(self, point_cloud: np.ndarray, num_points: int) -> np.ndarray:
        sampled_indices = np.random.choice(
            point_cloud.shape[0], num_points, replace=False
        )
        return point_cloud[sampled_indices]
