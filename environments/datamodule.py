from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal

import lightning as L
from torch.utils.data import DataLoader

from transforms.base_transform import init_transforms

if TYPE_CHECKING:
    from environments.datasets.base_dataset import TrajectoryDataset
    from environments.specs import DataSpecs
    from transforms.base_transform import TransformPartialsDict


class TrajectoryDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: Callable[[TransformPartialsDict | None], TrajectoryDataset],
        batch_size: int,
        simulation=None,
        validation: Literal["sim"] | float | None = None,
        preprocess_transforms: TransformPartialsDict | None = None,
        cpu_transforms: TransformPartialsDict | None = None,
        cpu_batch_transforms: TransformPartialsDict | None = None,
        gpu_batch_transforms: TransformPartialsDict | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,  # swallow remaining kwargs that are used for config
    ):
        super().__init__()
        self.dataset: TrajectoryDataset = dataset(transforms=cpu_transforms)
        specs = self.dataset.specs

        self.cpu_batch_transform, specs = init_transforms(cpu_batch_transforms, specs)
        self.gpu_batch_transform, specs = init_transforms(gpu_batch_transforms, specs)
        self._specs = specs

        self.batch_size = batch_size
        self.validation = validation
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def train_dataloader(self) -> Any:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> Any:
        return []

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return self.cpu_batch_transform(batch)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return self.gpu_batch_transform(batch)

    def _simulation_dataloader(self):
        simulation = SimulationDataset()
        dataloader = DataLoader(
            simulation,
            batch_size=None,  # disables automatic batching. in isaac, we can batch inside the simulation
            prefetch_factor=0,  # we cannot prefetch from the simulation
            persistent_workers=True,  # shutting down and restarting simulation is expensive
        )
        return dataloader
