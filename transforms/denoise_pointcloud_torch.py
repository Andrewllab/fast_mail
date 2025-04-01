import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph


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

        radius_graph = RadiusGraph(r=self.radius, max_num_neighbors=self.nb_points, loop=False) 
        pc = radius_graph(pc)

        row, col = pc.edge_index
        # https://pytorch.org/docs/stable/generated/torch.bincount.html
        deg = torch.bincount(row, minlength=pc.pos.size(0))

        mask = deg >= self.nb_points
        idx = torch.nonzero(mask, as_tuple=False).squeeze()

        pc.pos = pc.pos[idx]
        if pc.x is not None:
            pc.x = pc.x[idx]
        if pc.batch is not None:
            pc.batch = pc.batch[idx]

        return pc
