import logging
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from lightning import Callback, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from utils.conf import delete_keys_recursively, setup_resolvers
from utils.instantiators import instantiate_callbacks, instantiate_loggers
from utils.logging import configure_logging
from utils.seeding import get_rng, manual_seed
from utils.torch_conf import configure_torch
from utils.wandb import init_wandb

if TYPE_CHECKING:
    from environments.datasets.base_dataset import TrajectoryDataset

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.python_logging)

    # init wandb first so we can log any info or errors from instantiating dataset and model
    run = init_wandb(cfg)

    # recursively delete any "name" fields in config dictionary
    # we want these to be saved to WandB but we don't want them for instantiation
    cfg = delete_keys_recursively(cfg, ["name"])

    # configure torch, e.g. set_float32_matmul_precision
    configure_torch(cfg.get("torch"))

    # seeding
    rng = get_rng(cfg)
    seed = rng.integers(np.iinfo(np.uint32).max)
    log.info(f"Seeding pytorch, numpy, random, and workers with seed {seed}")
    seed_everything(seed, workers=True)

    # instantiate dataset
    dataset: TrajectoryDataset = hydra.utils.instantiate(cfg.data.dataset)
    dataloader = DataLoader(dataset, **cfg.dataloader, shuffle=True, drop_last=True)

    # instantiate agent
    # use "object" conversion strategy to avoid converting specs, which are a
    # dataclass, to a DictConfig
    agent: LightningModule = hydra.utils.instantiate(
        cfg.agent, specs=dataset.specs, _convert_="object"
    )

    log.debug("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.debug("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, _target_=Trainer, callbacks=callbacks, logger=logger
    )

    # unless disabled, create a simulation for validation during training
    val_dataloader = ()
    if not cfg.disable_validation:
        sim = hydra.utils.instantiate(cfg.data.simulation)
        sim_dataloader = DataLoader(sim, **cfg.sim_dataloader)
        val_dataloader += (sim_dataloader,)

    trainer.fit(agent, dataloader, *val_dataloader)

    log.info("Training done")
    run.finish()


if __name__ == "__main__":
    setup_resolvers()
    main()
