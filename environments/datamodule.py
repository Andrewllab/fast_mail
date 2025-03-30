from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Literal

import lightning as L
import torch
from torch.utils.data import DataLoader

from transforms.base_transform import init_transforms

if TYPE_CHECKING:
    from environments.datasets.base_dataset import DeviceType, TrajectoryDataset
    from environments.specs import DataSpecs
    from transforms.base_transform import TransformPartialsDict

log = logging.getLogger(__name__)


class TrajectoryDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: Callable,
        batch_size: int,
        device: DeviceType = "disk",
        simulation=None,
        validation: Literal["sim"] | float | None = None,
        preprocess_transforms: TransformPartialsDict | None = None,
        cpu_transforms: TransformPartialsDict | None = None,
        cpu_batch_transforms: TransformPartialsDict | None = None,
        gpu_batch_transforms: TransformPartialsDict | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self._dataset = dataset
        self._simulation = simulation
        self._preprocess_transforms = preprocess_transforms
        self._cpu_transforms = cpu_transforms
        self._cpu_batch_transforms = cpu_batch_transforms
        self._gpu_batch_transforms = gpu_batch_transforms

        if device not in ("disk", "cpu") and (
            cpu_transforms is not None or cpu_batch_transforms is not None
        ):
            raise ValueError(
                f"CPU transforms are not supported when dataset is stored on GPU."
            )

        self.batch_size = batch_size
        self._device = device
        self.validation = validation
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str) -> None:

        log.debug("Instantiating dataset...")
        self.dataset: TrajectoryDataset = self._dataset(
            device=self._device,
            transforms=self._cpu_transforms,
            preprocess_transforms=self._preprocess_transforms,
        )
        specs = self.dataset.specs

        log.debug("Instantiating cpu batch transforms...")
        self.cpu_batch_transform, specs = init_transforms(
            self._cpu_batch_transforms, specs
        )
        log.debug("Instantiating gpu batch transforms...")
        self.gpu_batch_transform, specs = init_transforms(
            self._gpu_batch_transforms, specs
        )
        self._specs = specs

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
        # TODO: how to disable validation even though it is implemented?
        if self.validation is None:
            log.warning(
                "Datamodule was asked to generate validation dataloaders, but validation mode is None! Returning empty list..."
            )
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
