import logging
from pathlib import Path

import hydra
import rootutils
import torch
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from agents.base_agent import BaseAgent
from environments.datamodule import TrajectoryDataModule
from loggers.wandb import resolve_checkpoint, update_wandb_config
from utils.conf import (
    delete_keys_recursively,
    merge_data_configs,
    patch_load_from_checkpoint,
    setup_resolvers,
)
from utils.instantiators import (
    instantiate_callbacks,
    instantiate_datamodule,
    instantiate_loggers,
)
from utils.legacy import patch_legacy_configs
from utils.logging import configure_logging, log_exception_and_finish_wandb
from utils.torch_conf import configure_torch

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs")
@log_exception_and_finish_wandb
def test(cfg: DictConfig) -> None:
    # resolve the entire config to catch any errors early
    OmegaConf.resolve(cfg)

    configure_logging(cfg.python_logging)

    # init wandb first so we can log any info or errors from instantiating dataset and model
    log.debug("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))
    train_tags = None
    train_notes = None

    agent_cfg = cfg.get("agent", {})
    data_cfg = cfg.get("data", {})
    checkpoint_cfg = cfg.get("checkpoint", {})
    if checkpoint := resolve_checkpoint(checkpoint_cfg):
        # load agent config and specs from checkpoint
        run, train_cfg, checkpoint_paths = checkpoint
        if isinstance(run, Path):
            run_name = run.name
        else:
            # wandb Api Run object
            run_name = run.id
            train_tags = run.tags
            train_notes = run.notes

        # use the first checkpoint to initialize the agent, if multiple are given
        epoch, checkpoint_path = checkpoint_paths[0]
        log.info(f"Loading checkpoint after epoch {epoch} of run {run_name}...")

        # merge the agent and data configs, with the current config taking precedence
        agent_cfg = OmegaConf.merge(train_cfg.agent, agent_cfg)
        data_cfg = merge_data_configs(train_cfg.data, data_cfg)

        agent_cfg = patch_legacy_configs(agent_cfg)
        data_cfg = patch_legacy_configs(data_cfg)

        # modify the _target_ to point to the module's load_from_checkpoint method
        agent_cfg = patch_load_from_checkpoint(agent_cfg, checkpoint_path)

        # save the merged configs back to cfg so they can be logged to wandb
        with open_dict(cfg):
            cfg.agent = agent_cfg
            cfg.data = data_cfg
            cfg.train_platform = train_cfg.platform

    else:
        # This case is uncommon, but can be useful for e.g. testing transforms
        # without training a model.
        checkpoint_paths = []
        log.info("No checkpoint specified, testing with untrained model...")

    update_wandb_config(cfg, train_tags, train_notes)

    # configure torch, e.g. set_float32_matmul_precision
    configure_torch(cfg.get("torch"))

    # instantiate dataset
    datamodule: TrajectoryDataModule = instantiate_datamodule(data_cfg)

    # manually run prepare data and setup so we can use dataset specs for model creation
    log.debug("Preparing and setting up data...")
    datamodule.prepare_data(stage="test")
    datamodule.setup(stage="test")

    if not checkpoint:
        # without a checkpoint, get the hparams from the datamodule like in train.py
        # the primary use case for predicting without loading a checkpoint is
        # open-loop replay
        datamodule_hparams = dict(
            specs=datamodule.specs,
            normalizer=datamodule.normalizer,
            reverse_transform=datamodule.reverse_transform,
        )
    else:
        # normalizer and reverse_transform are loaded from checkpoint
        datamodule_hparams = dict(specs=datamodule.specs)

    # instantiate agent
    log.debug("Instantiating agent...")
    # recursively delete these fields in config dictionary
    # we want these to be saved to WandB but we don't want them for instantiation
    delete_keys_recursively(agent_cfg, ["name"])
    agent: BaseAgent = hydra.utils.instantiate(agent_cfg, **datamodule_hparams)

    if checkpoint:
        agent.checkpoint_metadata = {
            "run_name": run_name,
            "epoch": epoch,
        }

    log.debug("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    callbacks.extend(datamodule.get_callbacks())

    log.debug("Instantiating trainer...")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, _target_=Trainer, callbacks=callbacks, logger=logger
    )

    log.info("Starting testing loop...")
    trainer.test(agent, datamodule=datamodule)

    # if multiple checkpoints are given, test them all
    for epoch, checkpoint_path in checkpoint_paths[1:]:
        log.info(f"Loading checkpoint at epoch {epoch} of run {run_name}...")
        # this is adapted from LightningModule.load_from_checkpoint but we don't
        # want to re-instantiate the model
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        agent.load_state_dict(checkpoint["state_dict"], strict=agent.strict_loading)
        agent.checkpoint_metadata = {
            "run_name": run_name,
            "epoch": epoch,
        }

        log.info("Continuing testing...")
        trainer.test(agent, datamodule=datamodule)

    log.info("Testing loop completed.")

    datamodule.close()


if __name__ == "__main__":
    setup_resolvers()
    test()
