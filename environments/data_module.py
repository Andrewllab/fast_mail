from typing import TYPE_CHECKING, Any, Callable

import lightning as L
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from environments.datasets.base_dataset import TrajectoryDataset
    from transforms.base_transform import TransformPartialsDict


class TrajectoryDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: Callable[[TransformPartialsDict | None], TrajectoryDataset],
        batch_size: int,
        preprocess_transforms: TransformPartialsDict | None = None,
        cpu_transforms: TransformPartialsDict | None = None,
        cpu_batch_transforms: TransformPartialsDict | None = None,
        gpu_batch_transforms: TransformPartialsDict | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.dataset = dataset(transforms=cpu_transforms)

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
