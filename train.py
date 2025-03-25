import logging

import hydra
import numpy as np
import torch
import wandb
from lightning import Callback, LightningModule, Trainer, seed_everything
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from utils.conf import pop_names, setup_resolvers
from utils.instantiators import instantiate_callbacks, instantiate_loggers
from utils.logging import configure_logging
from utils.seeding import get_rng, manual_seed
from utils.torch_conf import configure_torch

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    configure_logging(cfg.python_logging)

    with open_dict(cfg):
        notes = cfg.wandb.pop("notes", None)

    run = wandb.init(
        project="3d-sim2real",
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),  # type: ignore
        sync_tensorboard=True,  # auto-upload any values logged to tensorboard
        save_code=True,  # save script used to start training, git commit, and patch
        reinit=True,  # required for hydra sweeps with default launcher
        tags=cfg.wandb.get("tags"),
        notes=notes,
    )

    # recursively pop any "name" fields in config dictionary
    # we want these to be saved to WandB but we don't want them for instantiation
    cfg = pop_names(cfg)

    # configure torch, e.g. set_float32_matmul_precision
    configure_torch(cfg.get("torch", {}))

    # seeding
    rng = get_rng(cfg)
    seed = rng.integers(np.iinfo(np.uint32).max)
    log.info(f"Seeding pytorch, numpy, random, and workers with seed {seed}")
    seed_everything(seed, workers=True)

    # instantiate dataset
    dataset = hydra.utils.instantiate(cfg.dataset.dataset)
    dataloader = DataLoader(
        dataset, **cfg.dataloader, shuffle=True, pin_memory=True, drop_last=True
    )

    # instantiate agent
    agent: LightningModule = hydra.utils.instantiate(cfg.agent, dataset=dataset)

    param_count = sum(p.numel() for p in agent.parameters())
    log.info(f"Model parameter count: {param_count}")

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
        sim = hydra.utils.instantiate(cfg.dataset.simulation)
        sim_dataloader = DataLoader(sim, **cfg.sim_dataloader)
        val_dataloader += (sim_dataloader,)

    trainer.fit(agent, dataloader, *val_dataloader)

    log.info("Training done")
    run.finish()


if __name__ == "__main__":
    setup_resolvers()
    main()
