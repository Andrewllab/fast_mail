from torch_geometric.data import Data
from torch_geometric.nn import fps


from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform


class DownsampleFPSPointCloud(Transform):
    def __init__(
            self,
            specs: DataSpecs,
            num_samples: int = 512,
            ) -> None:

        self.num_samples = num_samples
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

        num_points = pc.pos.size(0)
        ratio = self.num_samples/num_points

        # https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.pool.fps.html
        fps_idx = fps(pc.pos, pc.batch, ratio=ratio) # TODO: Do we need the batch handling??

        sampled_pos = pc.pos[fps_idx]
        sampled_x = pc.x[fps_idx]
        sampled_batch = pc.batch[fps_idx]

        return Data(pos=sampled_pos, x=sampled_x, batch=sampled_batch)