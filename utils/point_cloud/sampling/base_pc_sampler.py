import numpy as np
import abc


class BasePointCloudSampler(abc.ABC):
    def sample(self, point_cloud: np.ndarray, num_points: int) -> np.ndarray:
        if point_cloud.shape[0] <= num_points:
            return point_cloud

        return self._sample(point_cloud, num_points)

    @abc.abstractmethod
    def _sample(self, point_cloud: np.ndarray, num_points: int) -> np.ndarray:
        pass
