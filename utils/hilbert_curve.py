import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve


def reorder_point_cloud_with_hilbert_curve(point_cloud):
    """
    Reorder a point cloud based on the Hilbert curve.

    Args:
        point_cloud (numpy.ndarray): A numpy array of shape (Batch, number, 3).

    Returns:
        numpy.ndarray: Reordered point cloud with the same shape as the input.
    """

    batch_size, num_points, _ = point_cloud.shape
    reordered_point_cloud = np.zeros_like(point_cloud)

    hilbert_resolution = 3
    hilbert_max_index = 2 ** hilbert_resolution - 1

    # Initialize Hilbert curve
    hilbert_curve = HilbertCurve(hilbert_resolution, 3)

    for b in range(batch_size):
        # Normalize points to [0, 2^resolution) range
        min_coords = point_cloud[b].min(axis=0)
        max_coords = point_cloud[b].max(axis=0)
        normalized_points = (point_cloud[b] - min_coords) / (max_coords - min_coords)

        hilbert_points = (normalized_points * hilbert_max_index).astype(int)

        hilbert_indices = hilbert_curve.distances_from_points(hilbert_points, match_type=True)

        # Reorder based on Hilbert indices
        sorted_indices = np.argsort(hilbert_indices)
        reordered_point_cloud[b] = point_cloud[b][sorted_indices]

    return reordered_point_cloud


# Example usage
if __name__ == "__main__":
    # Generate a random batch of point clouds
    batch_size = 2
    num_points = 100
    point_cloud = np.random.rand(batch_size, num_points, 3)

    # Reorder the point cloud
    reordered_point_cloud = reorder_point_cloud_with_hilbert_curve(point_cloud)
    print("Original Point Cloud:\n", point_cloud)
    print("Reordered Point Cloud:\n", reordered_point_cloud)
