from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Literal

import hydra
import lightning as L
import torch.nn as nn
from gymnasium import Wrapper
from gymnasium.wrappers import RecordVideo
from omegaconf import DictConfig
from tensordict import NonTensorData, TensorDict, is_leaf_nontensor
from torch import device
from torch.utils.data import DataLoader, Subset, random_split

from environments.base_dataset import DeviceType, TrajectoryDataset
from environments.collate import update_collate_fn_map
from environments.gym_env_dataset import GymEnvDataset
from environments.specs import DataSpecs
from transforms.base_transform import (
    Compose,
    NormalizingTransform,
    ReversibleTransform,
    Sequential,
    TransformPartialsDict,
    init_transforms,
)

log = logging.getLogger(__name__)


class TrajectoryDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: Callable | None = None,
        batch_size: int | None = None,
        device: DeviceType = "disk",
        preprocess_transforms: TransformPartialsDict | None = None,
        cpu_transforms: TransformPartialsDict | None = None,
        cpu_batch_transforms: TransformPartialsDict | None = None,
        gpu_batch_transforms: TransformPartialsDict | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        eval_mode: Literal["env", "dataset"] | float | None = None,
        env_dataset: DictConfig | None = None,
    ):
        super().__init__()
        self._dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self._preprocess_transforms = preprocess_transforms
        self._cpu_transforms = cpu_transforms
        self._cpu_batch_transforms = cpu_batch_transforms
        self._gpu_batch_transforms = gpu_batch_transforms
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.eval_mode = eval_mode
        self._env = env_dataset

        if device not in ("disk", "cpu") and (
            cpu_transforms is not None or cpu_batch_transforms is not None
        ):
            raise ValueError(
                f"CPU transforms are not supported when dataset is stored on GPU."
            )

        self.dataset: TrajectoryDataset | Subset | None = None
        self.eval_dataset: TrajectoryDataset | Subset | None = None
        self._specs: DataSpecs | None = None

        self.env: GymEnvDataset | None = None

    def setup(self, stage: str) -> None:

        if stage == "fit" or self.eval_mode == "dataset":
            # we need the dataset if we are training or if we are evaluating
            # on some complete dataset

            if stage == "fit" and self.eval_mode == "dataset":
                raise ValueError(
                    "Eval mode `dataset` is for evaluating on a complete dataset. To evaluate on a subset of the dataset, use a float value between 0 and 1."
                )

            self._instantiate_dataset()

            if isinstance(self.eval_mode, float) and self.eval_dataset is None:
                # if we are using a subset of the training dataset for evaluation,
                # we also need to split it first
                training = 1 - self.eval_mode
                log.debug(
                    f"Using {self.eval_mode * 100}% of the dataset for evaluation (validation/testing/prediction) and {training * 100}% for training."
                )
                self.dataset, self.eval_dataset = random_split(
                    self.dataset, [training, self.eval_mode]
                )
            elif self.eval_mode == "dataset":
                # if we are using a complete dataset for evaluation, we need to
                # set it here
                self.eval_dataset = self.dataset

        elif stage in ("test", "predict") and isinstance(self.eval_mode, float):
            raise ValueError(
                "Eval mode with a float means evaluating on a subset of the data, and is only supported for training."
            )

        elif stage in ("validate", "test", "predict") and self.eval_mode is None:
            raise ValueError(
                "Eval mode is None, but validate/test/predict stage was called. Please set eval_mode to a valid value."
            )

        if self.eval_mode == "env":
            # we need the environment if we are evaluating on an environment
            # we instantiate now to catch any errors before training starts
            self._instantiate_env_dataset()

    def _instantiate_dataset(self) -> None:
        if self.dataset is None:
            if self._dataset is None:
                raise ValueError("Dataset is not specified. Please provide a dataset.")
            log.debug("Instantiating dataset...")
            self.dataset = self._dataset(
                device=self.device,
                transforms=self._cpu_transforms,
                preprocess_transforms=self._preprocess_transforms,
            )
            assert isinstance(self.dataset, TrajectoryDataset)
            specs = self.dataset.specs
            self.preprocess_transforms = self.dataset.preprocess_transforms
            self.cpu_transforms = self.dataset.transform

            log.debug("Instantiating cpu batch transforms...")
            self.cpu_batch_transform, specs = init_transforms(
                self._cpu_batch_transforms, specs
            )
            log.debug("Instantiating gpu batch transforms...")
            self.gpu_batch_transform, specs = init_transforms(
                self._gpu_batch_transforms, specs
            )
            self._specs = specs

            # add support for collating TensorDicts and torch geometric data in
            # torch DataLoader
            # do this here to ensure it is run on every node
            update_collate_fn_map()

    def _instantiate_env_dataset(self) -> None:
        if self.env is None:
            if self._env is None:
                raise ValueError(
                    "Evaluation environment is not specified. Please provide an environment dataset."
                )
            log.debug("Instantiating environment...")
            # padding DictConfig avoids importing simulation modules until they are needed
            self.env = hydra.utils.instantiate(self._env, _partial_=False)
            specs = self.env.specs

            # Move any transforms that would normally be in the dataset (i.e.
            # preprocessing and cpu_transform) into cpu_batch_transforms. Any
            # transforms that only work in preprocessing should implement a
            # no-op __call__ method.
            preprocess_transforms, specs = init_transforms(
                self._preprocess_transforms, specs
            )
            cpu_transforms, specs = init_transforms(self._cpu_transforms, specs)

            # Filter out normalizing transforms, since they require running
            # preprocessing first, but they also cannot have a no-op __call__
            # method. The environment doesn't emit actions anyway (only
            # observations), so we don't care.
            preprocess_transforms = [
                t
                for t in preprocess_transforms
                if not isinstance(t, NormalizingTransform)
            ]

            log.debug("Instantiating cpu batch transforms for environment...")
            cpu_batch_transform, specs = init_transforms(
                self._cpu_batch_transforms, specs
            )

            log.debug(
                "Prepending preprocess and cpu transforms to cpu batch transforms for gym environment..."
            )
            env_cpu_batch_transform = (
                preprocess_transforms + list(cpu_transforms) + list(cpu_batch_transform)
            )
            self.env_cpu_batch_transform = Compose(*env_cpu_batch_transform)

            log.debug("Instantiating gpu batch transforms for environment...")
            self.env_gpu_batch_transform, specs = init_transforms(
                self._gpu_batch_transforms, specs
            )

            # if we have both a dataset and an environment, we need to check if
            # they have the same specs
            if self._specs is None:
                self._specs = specs
            elif specs != self.specs:
                raise ValueError(
                    "Specs of training dataset and environment dataset do not match. Please check your transforms."
                )

    @property
    def specs(self) -> DataSpecs:
        if self._specs is None:
            raise ValueError(
                "Specs are not available until the datamodule has been set up."
            )
        return self._specs

    @property
    def normalizer(self) -> Sequential | None:
        """Get a transform that normalizes the actions according to arbitrary
        dataset statistics collected during preprocessing. This transform (and
        its state) must be stored in the agent to be properly checkpointed.
        """
        if self.dataset is None:
            # Only a dataset emits actions, whereas environments do not.
            # Therefore we only need to worry about normalizing transforms on
            # the actions when we have a dataset.
            return None

        # we only check the preprocess transforms, because normalizers must be
        # in preprocessing so they can collect stats about the whole dataset
        normalizers = [
            t for t in self.preprocess_transforms if isinstance(t, NormalizingTransform)
        ]
        # normalizers have state, therefore they must inherit from nn.Module, so
        # we have to use Sequential and not Compose
        return Sequential(*normalizers) if normalizers else None

    @property
    def reverse_transform(self) -> Compose | None:
        """Get a transform that reverses any reversible transforms or
        normalizing applied to the actions. This transform must be stored in
        the agent to be properly checkpointed.
        """
        if self.dataset is None:
            # If we are being asked for a reverse transform, that means we are
            # not loading a checkpoint. Since we have an environment and no
            # dataset, this probably means we're doing open-loop replay. In
            # this case, we can't reverse any transforms because we don't know
            # what transforms were applied to the replay dataset.
            return None

        transforms = itertools.chain(
            self.preprocess_transforms,
            self.cpu_transforms,
            self.cpu_batch_transform,
            self.gpu_batch_transform,
        )

        # note: this isinstance check also includes any normalizing transforms,
        # which also need to be reversed
        reversible_transforms = [
            t for t in transforms if isinstance(t, ReversibleTransform)
        ]

        cls = (
            Sequential
            if any(isinstance(t, nn.Module) for t in reversible_transforms)
            else Compose
        )

        return cls(*reversible_transforms) if reversible_transforms else None

    def get_callbacks(self) -> list[L.Callback]:
        """This function is where I hide all the dirtiest parts of my code."""

        callbacks = []
        if self.env is not None:
            # import these here to prevent circular import
            from callbacks.action_writer import ActionWriter
            from callbacks.video_metadata_writer import VideoMetadataWriter

            log.info("Adding ActionWriter callback for env dataset.")
            callbacks.append(ActionWriter(self.env))

            wrappers = []
            env = self.env.env
            # TODO: what about VectorWrapper?
            while isinstance(env, Wrapper):
                wrappers.append(env)
                env = env.env  # go one level deeper

            video_recorders = [w for w in wrappers if isinstance(w, RecordVideo)]
            if video_recorders:
                assert len(video_recorders) == 1
                log.info("Adding VideoMetadataWriter callback for RecordVideo wrapper.")
                callbacks.append(VideoMetadataWriter(video_recorders[0]))

        return callbacks

    def train_dataloader(self) -> Any:
        log.debug("Creating new training dataloader...")
        assert self.dataset is not None, "Dataset is not set."
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Any:
        return self._evaluation_dataloader("val")

    def test_dataloader(self) -> Any:
        return self._evaluation_dataloader("test")

    def predict_dataloader(self) -> Any:
        return self._evaluation_dataloader("predict")

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if (
            self.trainer.training
            or self.eval_mode == "dataset"
            or isinstance(self.eval_mode, float)
        ):
            return self.cpu_batch_transform(batch)
        elif self.eval_mode == "env":
            return self.env_cpu_batch_transform(batch)

    def transfer_batch_to_device(
        self, batch: Any, device: device, dataloader_idx: int
    ) -> Any:

        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)

        if isinstance(batch, TensorDict):
            # by default, NonTensorData (such as PyG Data objects) are not
            # moved to the GPU, so we have to do it manually
            for key, value in batch.items(
                include_nested=True, leaves_only=True, is_leaf=is_leaf_nontensor
            ):
                if isinstance(value, NonTensorData):
                    batch[key] = value.data.to(device)

        return batch

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if (
            self.trainer.training
            or self.eval_mode == "dataset"
            or isinstance(self.eval_mode, float)
        ):
            return self.gpu_batch_transform(batch)
        elif self.eval_mode == "env":
            return self.env_gpu_batch_transform(batch)

    def _evaluation_dataloader(self, stage: str):
        if self.eval_mode == "env":
            if self.env is None:
                raise ValueError(
                    "Environment dataloader requested, but no environment provided."
                )
            # we do not use multiprocessing for the environment dataloader, since
            # each worker would need to create its own set of environments
            return DataLoader(
                self.env,
                # disables automatic batching. we batch either inside a gymnasium
                # vecenv or inside the environment itself (e.g. isaacsim)
                batch_size=None,
                collate_fn=lambda x: x,  # we don't need to collate or convert individual tensordicts
                num_workers=0,  # this is the default, but we set it explicitly
            )
        elif self.eval_mode == "dataset" or isinstance(self.eval_mode, float):
            assert self.eval_dataset is not None
            return DataLoader(
                self.eval_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=False,
                drop_last=False,
            )
        elif self.eval_mode is None:
            log.warning(
                f"Datamodule was asked to generate dataloaders for {stage}, but evaluation mode is None! Returning empty list..."
            )
            return DataLoader([])
        else:
            raise ValueError(f"Invalid evaluation mode: {self.eval_mode}")

    def teardown(self, stage: str) -> None:
        log.debug(f"Called teardown in stage {stage}")
        if self.env is not None:
            self.env.teardown()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
