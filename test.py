import logging

import hydra
import numpy as np
import rootutils
import torch
from lightning import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from environments.datamodule import TrajectoryDataModule
from loggers.wandb import resolve_checkpoint, update_wandb_table
from utils.conf import (
    delete_keys_recursively,
    patch_load_from_checkpoint,
    setup_resolvers,
)
from utils.instantiators import (
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_loggers,
)
from utils.logging import configure_logging
from utils.torch_conf import configure_torch

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig) -> None:
    # resolve the entire config to catch any errors early
    OmegaConf.resolve(cfg)

    configure_logging(cfg.python_logging)

    # init wandb first so we can log any info or errors from instantiating dataset and model
    log.debug("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg)

    # configure torch, e.g. set_float32_matmul_precision
    configure_torch(cfg.get("torch"))

    # instantiate dataset
    datamodule: TrajectoryDataModule = instantiate_datamodule(cfg.data)

    # manually run prepare data and setup so we can use dataset specs for model creation
    log.debug("Instantiating datamodule...")
    datamodule.prepare_data()
    datamodule.setup(stage="test")

    # instantiate agent
    log.debug("Instantiating agent...")
    if checkpoint := resolve_checkpoint(cfg, use_artifact=True):
        # load agent config and specs from checkpoint
        train_cfg, checkpoint_path = checkpoint
        agent_cfg = train_cfg.agent
        # modify the _target_ to point to the module's load_from_checkpoint method
        agent_cfg = patch_load_from_checkpoint(agent_cfg, checkpoint_path)
        datamodule_hparams = {}  # hparams are loaded from checkpoint
    else:
        agent_cfg = cfg.agent
        # without a checkpoint, get the hparams from the datamodule like in train.py
        datamodule_hparams = dict(
            specs=datamodule.specs,
            normalizer=datamodule.normalizer,
            reverse_transform=datamodule.reverse_transform,
        )
    # recursively delete these fields in config dictionary
    # we want these to be saved to WandB but we don't want them for instantiation
    delete_keys_recursively(agent_cfg, ["name"])
    agent: LightningModule = hydra.utils.instantiate(agent_cfg, **datamodule_hparams)

    log.debug("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.debug("Instantiating trainer...")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, _target_=Trainer, callbacks=callbacks, logger=logger
    )

    log.info("Starting testing loop...")
    trainer.test(agent, datamodule=datamodule)

    log.info("Testing loop completed.")

    # TODO: Make uploading evaluation videos to wandb optional
    log.info("Starting uploading evaluation videos to wandb.")
    update_wandb_table(cfg=cfg)
    log.info("Uploading evaluation videos to wandb completed.")

    # version = cfg.get("artifact_version", "latest") or "latest"
    # wandb.init(
    #     project=cfg.artifact_project,
    #     entity=cfg.artifact_entity,
    #     id=cfg.artifact_run_name,  # This is the original run ID from training
    #     resume="must"
    # )
    # recording_dir = Path(cfg.paths.recording_dir)
    # recording_paths = [
    #     str(p.resolve()) for p in recording_dir.rglob("*.mp4")
    # ]

    # episodes_by_ckpt = defaultdict(list)
    # for recording_path in recording_paths:
    #     # parts = path.stem.split("-")  # e.g., ['eval', 'v2', 'ep3']
    #     # if len(parts) < 3:
    #     #     continue
    #     # _, version, _ = parts
    #     episodes_by_ckpt[version].append(wandb.Video(recording_path, fps=30, format="mp4"))

    # # Check all logged history keys (metrics, tables, etc.)
    # keys = run.history_keys
    # print(keys)

    # # Or, more directly check if a specific key exists
    # if "evaluation" in keys:
    #     print("Table already exists!")

    # # TODO: this rewrites every time the table. Check if it exist and then only update it to prevent removing information from previous versions tests.
    # table = wandb.Table(columns=["checkpoint_version", "episodes"])

    # for version, video_list in sorted(episodes_by_ckpt.items()):
    #     table.add_data(version, video_list)

    # wandb.log({"evaluation": table})
    # wandb.finish()


if __name__ == "__main__":
    setup_resolvers()
    main()
