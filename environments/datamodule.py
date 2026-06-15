from __future__ import annotations

import functools
import itertools
import logging
import re
import shutil
from typing import Any, Literal

import hydra
import lightning as L
import torch
import torch.nn as nn
from hydra.errors import InstantiationException
from omegaconf import DictConfig, open_dict
from tensordict import NonTensorData, TensorDict, is_leaf_nontensor
from torch.utils.data import DataLoader, Subset, random_split

from environments.base_dataset import (
    TrajectoryDataset,
    TrajectorySubset,
    random_traj_split,
)
from environments.collate import update_collate_fn_map
from environments.gym_env_dataset import GymEnvDataset
from environments.specs import DataSpecs
from transforms.base_transform import (
    KEY_PATTERN,
    Compose,
    GpuExecutionWrapper,
    NormalizingTransform,
    ReversibleTransform,
    Sequential,
    TransformConstraint,
    TransformPartialsDict,
    get_transforms_config,
    init_transforms,
)
from utils.instantiators import get_dataset_class
from utils.paths import resolve_path

log = logging.getLogger(__name__)


class EmptyPointCloudError(Exception):
    pass


class EmptyTrajectoryError(Exception):
    pass


class TrajectoryDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        action_seq_len: int,
        obs_seq_len: int,
        batch_size: int | None,
        cpu_transforms: TransformPartialsDict | None = None,
        cpu_batch_transforms: TransformPartialsDict | None = None,
        gpu_batch_transforms: TransformPartialsDict | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: int | None = None,
        eval_mode: Literal["env", "dataset"] | float | None = None,
        eval_traj_subset: bool = True,
        env: DictConfig | None = None,
        **preprocess_cfgs: DictConfig,
    ):
        super().__init__()
        self.dataset_cfg = dataset
        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len
        self.batch_size = batch_size
        self._cpu_transforms = cpu_transforms
        self._cpu_batch_transforms = cpu_batch_transforms
        self._gpu_batch_transforms = gpu_batch_transforms
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.eval_mode = eval_mode
        self.eval_traj_subset = eval_traj_subset
        self.env_cfg = env

        # sort by number of "pre" prefixes
        preprocess_cfgs = dict(
            sorted(
                preprocess_cfgs.items(),
                key=lambda kv: self.preprocess_keyfunc(kv[0]),
            )
        )
        # filter out empty preprocessing steps
        self.preprocess_cfgs = {
            key: cfg
            for key, cfg in preprocess_cfgs.items()
            if get_transforms_config(cfg)
        }

        self.dataset: TrajectoryDataset | TrajectorySubset | Subset | None = None
        self.eval_dataset: TrajectoryDataset | TrajectorySubset | Subset | None = None
        self._specs: DataSpecs | None = None

        self.env: GymEnvDataset | None = None

        # add support for collating TensorDicts and torch geometric data in
        # torch DataLoader
        update_collate_fn_map()

    @staticmethod
    def preprocess_keyfunc(key: str) -> int:
        # Matches strings like "process", "preprocess", "prepreprocess", etc.
        match = re.fullmatch(r"^(pre)*process$", key)
        if not match:
            raise ValueError(f"Invalid key format: {key!r}")
        return key.count("pre")

    def prepare_data(self, stage: str | None = None) -> None:
        # TODO: skip preprocess if it's already been run in this process before

        if stage is None:
            assert self.trainer is not None
            stage = self.trainer.state.fn
        needs_dataset, needs_env = self.needs_setup(stage, self.eval_mode)

        if not needs_dataset or self.dataset is not None:
            return

        log.info("Preparing data...")

        # create minimal transform configs for testing if existing preprocessed
        # data has matching transforms
        transform_cfgs = {
            key: get_transforms_config(cfg) for key, cfg in self.preprocess_cfgs.items()
        }

        # accumulate the transform ListConfigs in reverse order, i.e. the first
        # step we consider is the last processing step, and therefore has the
        # transforms of all previous steps applied before it
        cumulative_transform_cfgs = dict(
            zip(
                transform_cfgs.keys(),
                reversed(list(itertools.accumulate(reversed(transform_cfgs.values())))),
            )
        )

        # check which preprocessing steps need to be done, if any
        step = -1
        for step, (key, preprocess_cfg) in enumerate(self.preprocess_cfgs.items()):

            # remove this key, as it's only relevant for verification and the
            # datasets don't expect it as an argument
            overwrite = preprocess_cfg.pop("overwrite", False)

            # the dataset config is made by removing all transform configs,
            # including those that are not partials or are set to None
            dataset_cfg = {
                key: value
                for key, value in preprocess_cfg.items()
                if not KEY_PATTERN.fullmatch(key)
            }

            DatasetCls = get_dataset_class(dataset_cfg)
            root_dir = resolve_path(dataset_cfg["root_dir"])

            try:
                # TODO: add some kind of file manifest for a more robust check
                dataset = hydra.utils.instantiate(
                    dataset_cfg,
                    _target_=DatasetCls,
                    _partial_=False,
                    action_seq_len=self.action_seq_len,
                    obs_seq_len=self.obs_seq_len,
                )

            except (FileNotFoundError, InstantiationException) as e:
                if not (
                    isinstance(e, FileNotFoundError)
                    or isinstance(e.__cause__, FileNotFoundError)
                ):
                    raise e

                if root_dir.is_dir() and any(root_dir.iterdir()):
                    if overwrite:
                        log.warning(
                            f"Preprocessed directory {root_dir} is not empty. Overwriting..."
                        )
                        shutil.rmtree(root_dir)
                    else:
                        raise FileExistsError(
                            f"Preprocessed directory {root_dir} is not empty (perhaps an incomplete earlier preprocessing run). Set overwrite=True to overwrite."
                        )

                # Dataset not found. Continue search with next processing step.
                continue

            # check if the existing preprocessed data has the same transforms
            old_config = dataset.preprocess_transforms_config
            new_config = cumulative_transform_cfgs[key]

            if old_config != new_config:
                if overwrite:
                    log.warning(
                        f"Preprocess config does not match existing preprocessed data in {root_dir}. Overwriting..."
                    )
                    log.warning(
                        f"Old config:\n{old_config}\n\nNew config:\n{new_config}"
                    )
                    shutil.rmtree(root_dir)
                    continue
                else:
                    log.warning(
                        f"Old config:\n{old_config}\n\nNew config:\n{new_config}"
                    )
                    raise ValueError(
                        f"Preprocess config does not match existing preprocessed data in {root_dir}. Set overwrite=True to overwrite."
                    )

            log.info(
                f"Found {key}ed data in {root_dir} ({dataset.n_trajectories} trajectories)."
            )
            break

        else:
            # need to preprocess raw data
            step += 1  # increment step to include the final preprocessing step

            dataset_cfg = self.dataset_cfg.copy()
            DatasetCls = get_dataset_class(dataset_cfg)
            dataset = hydra.utils.instantiate(
                dataset_cfg,
                _target_=DatasetCls,
                _partial_=False,
                action_seq_len=self.action_seq_len,
                obs_seq_len=self.obs_seq_len,
            )
            log.info(
                f"Found raw data in {dataset.root_dir} ({dataset.n_trajectories} trajectories)."
            )

        # take only the preprocessing steps that need to be done, and reverse their order
        preprocess_cfgs = dict(reversed(list(self.preprocess_cfgs.items())[:step]))

        for step, (key, preprocess_cfg) in enumerate(preprocess_cfgs.items()):

            # the dataset config is made by removing all transform configs,
            # including those that are not partials or are set to None
            dataset_cfg = {
                key: value
                for key, value in preprocess_cfg.items()
                if not KEY_PATTERN.fullmatch(key)
            }

            DatasetCls = get_dataset_class(dataset_cfg)
            root_dir = resolve_path(dataset_cfg["root_dir"])
            log.info(f"Running {key}ing and saving to {root_dir}...")

            root_dir.mkdir(parents=True, exist_ok=True)

            specs = dataset.specs
            transforms, specs = init_transforms(preprocess_cfg, specs)
            assert isinstance(transforms, Compose)
            assert specs == transforms[-1].specs

            # Wrap any transforms that need to be executed on the GPU.
            # First we create a copy of the transforms list, since we don't want
            # to save the wrapped transforms as metadata, only the originals.
            _transforms = list(transforms)
            for j, transform in enumerate(_transforms):
                if TransformConstraint.GPU_ONLY in transform.constraints:
                    _transforms[j] = GpuExecutionWrapper(transform)

            n_trajs = dataset.n_trajectories
            for idx in range(n_trajs):
                log.debug(f"Loading trajectory #{idx + 1} of {n_trajs}...")
                traj = dataset.get_trajectory(idx)
                log.debug(
                    f"{key.title()}ing trajectory "
                    + (f"named {traj['name']} " if "name" in traj else "")
                    + f"from {traj['path']}..."
                )
                trajs = [traj]

                for transform in _transforms:
                    next_trajs = []
                    for traj in trajs:
                        try:
                            transformed = transform.call_trajectory(traj)
                        except EmptyPointCloudError as e:
                            log.warning(
                                "Skipping trajectory %sfrom %s due to empty point cloud after %s: %s",
                                f"named {traj['name']} " if "name" in traj else "",
                                traj["path"],
                                transform.__class__.__name__,
                                e,
                            )
                            continue

                        if isinstance(transformed, list):
                            next_trajs.extend(transformed)
                        else:
                            next_trajs.append(transformed)

                    trajs = next_trajs

                for traj in trajs:
                    DatasetCls.save_trajectory(traj, root_dir, specs)

            # accumulate all transforms applied in all preprocessing steps so far
            cumulative_transforms = dataset.preprocess_transforms + transforms

            cumulative_transform_cfg = cumulative_transform_cfgs[key]
            DatasetCls.save_metadata(
                root_dir, cumulative_transforms, cumulative_transform_cfg
            )

            log.info(f"Finished {key}ing (data saved to {root_dir}).")

            if step < len(preprocess_cfgs) - 1:
                # create dataset object to act as the source for the next preprocessing step
                dataset = hydra.utils.instantiate(
                    dataset_cfg,
                    _target_=DatasetCls,
                    _partial_=False,
                    action_seq_len=self.action_seq_len,
                    obs_seq_len=self.obs_seq_len,
                )

    def setup(self, stage: str) -> None:

        log.debug(f"Setting up Datamodule for {stage} stage")

        needs_dataset, needs_env = self.needs_setup(stage, self.eval_mode)

        if needs_dataset and self.dataset is None:
            self._instantiate_dataset()

            if isinstance(self.eval_mode, float):
                # if we are using a subset of the training dataset for evaluation,
                # we also need to split it first
                training = 1 - self.eval_mode
                log.info(
                    f"Using {self.eval_mode * 100}% of the trajectories in the dataset for evaluation (validation/testing/prediction) and {training * 100}% for training."
                )
                if self.eval_traj_subset:
                    self.dataset, self.eval_dataset = random_traj_split(
                        self.dataset, [training, self.eval_mode]
                    )
                    log.debug(
                        f"The following {self.eval_dataset.n_trajectories} trajectories will be used for evaluation:\n{self.eval_dataset.traj_indices} "
                    )
                else:
                    self.dataset, self.eval_dataset = random_split(
                        self.dataset, [training, self.eval_mode]
                    )

            elif self.eval_mode == "dataset":
                # if we are using a complete dataset for evaluation, we need to
                # set it here
                self.eval_dataset = self.dataset

        if needs_env and self.env is None:
            # we need the environment if we are evaluating on an environment
            # we instantiate now to catch any errors before training starts
            self._instantiate_env_dataset()

    @staticmethod
    def needs_setup(stage: str, eval_mode: str | float | None) -> tuple[bool, bool]:
        """Check if the datamodule needs to be set up for the given stage and
        eval_mode.

        Returns a tuple of booleans indicating whether the dataset and/or
        environment need to be set up.
        """
        if stage == "fit" and eval_mode == "dataset":
            raise ValueError(
                "Eval mode `dataset` is for evaluating on a complete dataset. To evaluate on a subset of the dataset, use a float value between 0 and 1."
            )

        if stage in ("test", "predict") and isinstance(eval_mode, float):
            raise ValueError(
                "Eval mode with a float means evaluating on a subset of the data, and is only supported for training."
            )

        if stage in ("validate", "test", "predict") and eval_mode is None:
            raise ValueError(
                "Eval mode is None, but validate/test/predict stage was called. Please set eval_mode to a valid value."
            )

        # we need the dataset if we are training or if we are evaluating
        # on some complete dataset
        needs_dataset = stage == "fit" or eval_mode == "dataset"

        # TODO: maybe only if stage is validate/test/predict?
        needs_env = eval_mode == "env"

        return needs_dataset, needs_env

    def teardown(self, stage: str) -> None:
        log.debug(f"Tearing down Datamodule after {stage} stage")
        # TODO: maybe decide if the envs need to be destroyed and created for each validation
        if self.env is not None:
            self.env.teardown()

    def close(self) -> None:
        if self.env is not None:
            self.env.close()

    def _instantiate_dataset(self) -> None:
        log.info("Instantiating dataset...")

        # the dataset config is made by removing all transform configs,
        # including those that are not partials or are set to None
        if self.preprocess_cfgs:
            preprocess_cfg = list(self.preprocess_cfgs.values())[0]
            preprocess_cfg.pop("overwrite", False)
            dataset_cfg = {
                key: value
                for key, value in preprocess_cfg.items()
                if not KEY_PATTERN.fullmatch(key)
            }

        else:
            dataset_cfg = self.dataset_cfg.copy()

        DatasetCls = get_dataset_class(dataset_cfg)
        self.dataset = hydra.utils.instantiate(
            dataset_cfg,
            _target_=DatasetCls,
            _partial_=False,
            action_seq_len=self.action_seq_len,
            obs_seq_len=self.obs_seq_len,
            item_transforms=self._cpu_transforms,
        )

        assert isinstance(self.dataset, TrajectoryDataset)
        specs = self.dataset.specs
        log.debug(
            f"Dataset trajectories have the following lengths:\n{specs.traj_lengths}"
        )
        log.info(f"Dataset contains {len(self.dataset)} samples in total.")

        self.preprocess_transforms = self.dataset.preprocess_transforms
        self.cpu_transforms = self.dataset.item_transforms

        log.info("Instantiating cpu batch transforms...")
        self.cpu_batch_transform, specs = init_transforms(
            self._cpu_batch_transforms, specs
        )
        log.info("Instantiating gpu batch transforms...")
        self.gpu_batch_transform, specs = init_transforms(
            self._gpu_batch_transforms, specs
        )
        self._specs = specs

    def _instantiate_env_dataset(self) -> None:
        if self.env_cfg is None:
            raise ValueError(
                "Evaluation environment is not specified. Please provide an environment dataset."
            )
        log.info("Instantiating environment...")
        with open_dict(self.env_cfg):
            num_episodes = self.env_cfg.pop("num_episodes", None)

        env = hydra.utils.instantiate(self.env_cfg, _partial_=False)
        self.env = GymEnvDataset(env, num_episodes=num_episodes)
        specs = self.env.specs

        # Filter out transforms that should not be applied in the environment.
        # This includes transforms with constraints TRAJECTORY_ONLY or DATASET_ONLY,
        # as well as normalizing transforms (which are already embedded in the
        # agent anyway).
        preprocess_cfgs = {}
        DISALLOWED_CONSTRAINTS = {
            TransformConstraint.TRAJECTORY_ONLY,
            TransformConstraint.DATASET_ONLY,
        }
        for key, cfg in self.preprocess_cfgs.items():
            preprocess_cfg = {}
            for name, transform in cfg.items():
                # filter out non-partials
                if not isinstance(transform, functools.partial):
                    continue

                # filter out normalizing transforms
                # TODO: this is a legacy check, and NormalizingTransform should
                # be replaced with something less confusing.
                if issubclass(transform.func, NormalizingTransform):
                    continue

                # filter out transforms that cannot be applied outside of
                # preprocessing/dataset
                if set(transform.func.constraints) & DISALLOWED_CONSTRAINTS:
                    continue

                preprocess_cfg[name] = transform
            preprocess_cfgs[key] = preprocess_cfg

        # instantiate transforms from each preprocessing step separately to not mess
        # up ordering
        # reverse the order, so that e.g. prepreprocess comes before preprocess
        log.info("Instantiating preprocess transforms for environment...")
        preprocess_transforms = Compose(specs=specs)
        for cfg in reversed(preprocess_cfgs.values()):
            transforms, specs = init_transforms(cfg, specs)
            assert isinstance(transforms, Compose)
            preprocess_transforms += transforms

        log.info("Instantiating item transforms for environment...")
        cpu_transforms, specs = init_transforms(self._cpu_transforms, specs)
        preprocess_transforms += cpu_transforms

        log.info("Instantiating cpu batch transforms for environment...")
        cpu_batch_transform, specs = init_transforms(self._cpu_batch_transforms, specs)

        log.info("Instantiating gpu batch transforms for environment...")
        gpu_batch_transforms, specs = init_transforms(self._gpu_batch_transforms, specs)

        # Move any transforms that would normally be in the dataset (i.e.
        # preprocessing and cpu_transform) into cpu_batch_transforms. Any
        # transforms that only work in preprocessing should implement a
        # no-op __call__ method.
        if cpu_batch_transform:
            log.debug(
                "Prepending preprocess and cpu transforms to cpu batch transforms for gym environment..."
            )
            self.env_cpu_batch_transform = preprocess_transforms + cpu_batch_transform
            self.env_gpu_batch_transform = gpu_batch_transforms
        else:
            log.debug(
                "No cpu transforms found for gym environment. Prepending preprocess to gpu batch transforms for gym environment"
            )
            self.env_cpu_batch_transform = cpu_batch_transform
            self.env_gpu_batch_transform = preprocess_transforms + gpu_batch_transforms

        # if we have both a dataset and an environment, we need to check if
        # they have the same specs
        if self._specs is None:
            self._specs = specs
        elif specs != self.specs:
            raise ValueError(
                "Specs of training dataset and environment dataset do not match. Please check your transforms."
            )

        # Set transforms to eval mode, as they will be used for evaluation
        self.env_cpu_batch_transform.eval()
        self.env_gpu_batch_transform.eval()

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
            # return None
            transforms = itertools.chain(
                self.env_cpu_batch_transform,
                self.env_gpu_batch_transform,
            )
        else:
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
        # TODO: set up a system to register callbacks globally from anywhere

        callbacks = []
        if self.env is not None:
            # import these here to prevent circular import
            from callbacks.action_writer import ActionWriter
            from callbacks.video_metadata_writer import VideoMetadataWriter

            log.debug("Adding ActionWriter callback for env dataset.")
            callbacks.append(ActionWriter(self.env))

            if self.env.video_recorder is not None:
                log.debug(
                    "Adding VideoMetadataWriter callback for RecordVideo wrapper."
                )
                callbacks.append(VideoMetadataWriter(self.env.video_recorder))

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
            return DataLoader([])  # type: ignore
        else:
            raise ValueError(f"Invalid evaluation mode: {self.eval_mode}")

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
            # set the transforms to train or eval mode depending on whether
            # we are training or evaluating. We need to do this here because
            # the transforms are shared for train and eval dataloaders
            self.cpu_batch_transform.train(mode=self.trainer.training)
            return self.cpu_batch_transform(batch)
        elif self.eval_mode == "env":
            # For env evaluation, we always want the transforms to be in eval
            assert self.env_cpu_batch_transform.training is False
            return self.env_cpu_batch_transform(batch)

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:

        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)

        if isinstance(batch, TensorDict):
            # by default, NonTensorData (such as PyG Data objects) are not
            # moved to the GPU, so we have to do it manually
            for key, value in batch.items(
                include_nested=True, leaves_only=True, is_leaf=is_leaf_nontensor
            ):
                # check if the value is a NonTensorData and has a .to() method
                # e.g. strings do not
                if isinstance(value, NonTensorData) and hasattr(value.data, "to"):
                    batch[key] = value.data.to(device)

        if (self.dataset is not None) and isinstance(
            self.gpu_batch_transform, nn.Module
        ):
            self.gpu_batch_transform = self.gpu_batch_transform.to(device)
        elif self.env is not None and isinstance(
            self.env_gpu_batch_transform, nn.Module
        ):
            self.env_gpu_batch_transform = self.env_gpu_batch_transform.to(device)

        return batch

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if (
            self.trainer.training
            or self.eval_mode == "dataset"
            or isinstance(self.eval_mode, float)
        ):
            # set the transforms to train or eval mode depending on whether
            # we are training or evaluating. We need to do this here because
            # the transforms are shared for train and eval dataloaders
            self.gpu_batch_transform.train(mode=self.trainer.training)
            return self.gpu_batch_transform(batch)
        elif self.eval_mode == "env":
            assert self.env_gpu_batch_transform.training is False
            return self.env_gpu_batch_transform(batch)
