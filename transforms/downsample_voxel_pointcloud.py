from torch_geometric.data import Data
from torch_geometric.nn import max_pool, voxel_grid
from torch_geometric.transforms import T


from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform


class DownsampleVoxelPointCloud(Transform):
    def __init__(
            self,
            specs: DataSpecs,
            voxel_size: float = 0.1, # TODO: what is a suitable default value?
            start: int = None,
            end: int = None,
            ) -> None:

        self.voxel_size = voxel_size
        self.start = start
        self.end = end
        self._specs = specs
        self.transform = T.Cartesian(cat=False)

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
        # https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.pool.voxel_grid.html?highlight=voxel#torch_geometric.nn.pool.voxel_grid
        # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/mnist_voxel_grid.py
        
        cluster = voxel_grid(pc.pos, batch=pc.batch, size=self.voxel_size, start=self.start, end=self.end) # TODO: do we need the batch handling??

        pc.edge_attr = None # edge attributes are not relevant here
        pc = max_pool(cluster, pc, transform = self.transform)

        return pc
