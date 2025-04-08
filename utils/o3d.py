import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch.utils.dlpack


def torch_to_o3d(tensor: torch.Tensor) -> o3c.Tensor:
    """
    Convert a pytorch tensor (cpu or cuda) to an Open3D tensor.
    """
    return o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


def numpy_to_o3d(array: np.ndarray) -> o3c.Tensor:
    """
    Convert a numpy array to an Open3D tensor.
    """
    return o3c.Tensor.from_numpy(array)
