import logging
from typing import Sequence

from tensordict import NonTensorData
from torch_geometric.data import Batch, Data

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform
from utils.pyg import apply_index, batch2ptr, fps, update_batch_metadata

log = logging.getLogger(__name__)


class FpsSamplePointCloud(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        ratio: float | None = None,
        n_points: int | None = None,
        random_start: bool = True,
        pcd_keys: str | Sequence[str] = "pcd",
    ) -> None:

        if ratio is not None and n_points is not None:
            raise ValueError("Only one of ratio or n_points can be set.")
        elif ratio is None and n_points is None:
            raise ValueError("One of ratio or n_points must be set.")
        self.ratio = ratio
        self.n_points = n_points
        self.random_start = random_start

        if isinstance(pcd_keys, str):
            pcd_keys = [pcd_keys]
        else:
            pcd_keys = list(pcd_keys)
        self._pcd_keys = pcd_keys

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

    def _call_one(self, nt_data: NonTensorData) -> Data:
        data: Data = nt_data.data  # unpack NonTensorData wrapper around pyg Data object

        assert isinstance(data, Data)
        pos = data.pos
        assert pos is not None

        if isinstance(data, Batch):
            batch, ptr = data.batch, data.ptr
            assert batch is not None
            assert ptr is not None

            # verify that ptr is up to date
            if data.ptr[-1] != batch.shape[0]:
                ptr = batch2ptr(batch)

        else:
            ptr = None

        # pos: (B*N, 3)
        # idxs: (B*C)
        idxs = fps(
            pos,
            ptr=ptr,
            ratio=self.ratio,
            n_points=self.n_points,
            random_start=self.random_start,
        )

        data = apply_index(data, idxs)
        data = update_batch_metadata(data)

        return data

    def __repr__(self) -> str:
        if self.ratio is not None:
            args = f"ratio={self.ratio}"
        else:
            args = f"n_points={self.n_points}"
        return f"{self.__class__.__name__}({args})"
