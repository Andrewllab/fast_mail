import torch
from torch_geometric.data import Data
import torchvision.transforms.functional as F


from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform


class DownsampleRandomPointCloud(Transform):
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

        if self.num_samples >= num_points: # TODO: @Balazs, how to handle errors like this in best way? Pad with zeros or error?
            raise ValueError(f"Point cloud too small: trying to sample {self.num_samples}, but only {num_points} available.")
        
        idx = torch.randperm(num_points)[:self.num_samples]

        sampled_pc = pc.clone()
        sampled_pc.pos = pc.pos[idx]
        if pc.x is not None:
            sampled_pc.x = pc.x[idx]

        return sampled_pc
