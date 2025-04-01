import torch
from torch_geometric.data import Data
import torchvision.transforms.functional as F
import open3d as o3d


from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform


class DenoisePointCloud(Transform):
    def __init__(
            self,
            specs: DataSpecs,
            nb_points: int = 30,
            radius: float = 0.03
            ) -> None:

        self.nb_points = nb_points
        self.radius = radius
        self._specs = specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=[("obs", "pcd")],
                out_keys=[("obs", "pcd")]
                )]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, pc: Data) -> Data:

        pos = pc.pos.detach().cpu().numpy() # open3d expects numpy arrays
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos)

        # color handling
        if hasattr(pc, 'x') and pc.x is not None and pc.x.shape[1] >= 3:
            colors = pc.x[:, :3].detach().cpu().numpy()
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)

        cl, ind = pcd.remove_radius_outlier(nb_points=self.nb_points, radius=self.radius)
        inlier_cloud = pcd.select_by_index(ind)
        inlier_indices = torch.tensor(ind, dtype=torch.long)

        filtered_pos = torch.tensor(inlier_cloud.points, dtype=torch.float)
        new_pc = Data(pos=filtered_pos)

        # color handling
        if hasattr(pc, 'x') and pc.x is not None:
            new_pc.x = pc.x[inlier_indices]

        return pc
