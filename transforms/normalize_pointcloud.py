from __future__ import annotations

from typing import TYPE_CHECKING

from environments.specs import DataSpecs, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform

if TYPE_CHECKING:
    from torch_geometric.data import Data


class NormalizePointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        center: tuple[float, float, float] | None = None,
        scale: tuple[float, float, float] | None = None,
    ) -> None:
        self.center = center
        self.scale = scale

        self._pcd_keys = [
            key for key, spec in specs.obs.items() if isinstance(spec, PointCloudSpec)
        ]
        self._specs = specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(in_keys=[("obs", key)], out_keys=[("obs", key)])
            for key in self._pcd_keys
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, pcd: Data) -> Data:
        assert pcd.pos is not None

        if self.center is None:
            # center the point cloud at the origin
            center = pcd.pos[:, :3].mean(dim=-2, keepdim=True)
        pcd.pos[...] -= center

        if self.scale is None:
            # scale such that the maximum coordinate (in any dimension) is 1
            # this is equivalent to scaling the point cloud to fit in a unit cube
            max = (
                pcd.pos.abs()
                # can you believe that max does not take a tuple as a dim argument?
                .flatten(start_dim=-2)
                .max(dim=-1, keepdim=True)
                .values.unsqueeze(-1)
            )
            scale = 0.999999 / max
        pcd.pos[...] *= scale

        return pcd
