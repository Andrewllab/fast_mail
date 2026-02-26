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


@hydra.main(version_base=None, config_path="configs", config_name="test_real_pair")
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

    checkpoint_cfg = cfg.get("checkpoint_1", {})
    checkpoint = resolve_checkpoint(checkpoint_cfg)
    assert checkpoint is not None
    # load agent config and specs from checkpoint
    run, train_cfg, checkpoint_paths = checkpoint
    if isinstance(run, Path):
        run_name = run.name
    else:
        # wandb Api Run object
        run_name = run.id
        train_tags = run.tags
        train_notes = run.notes

    assert len(checkpoint_paths) == 1
    epoch, checkpoint_path = checkpoint_paths[0]
    metadata_1 = {
        "run_name": run_name,
        "epoch": epoch,
    }
    log.info(f"Loading checkpoint after epoch {epoch} of run {run_name}...")

    # merge the agent and data configs, with the current config taking precedence
    agent_cfg_1 = OmegaConf.merge(train_cfg.agent, agent_cfg)
    data_cfg_1 = merge_data_configs(train_cfg.data, data_cfg)

    agent_cfg_1 = patch_legacy_configs(agent_cfg_1)
    data_cfg_1 = patch_legacy_configs(data_cfg_1)

    # modify the _target_ to point to the module's load_from_checkpoint method
    agent_cfg_1 = patch_load_from_checkpoint(agent_cfg_1, checkpoint_path)

    # save the merged configs back to cfg so they can be logged to wandb
    with open_dict(cfg):
        cfg.agent = agent_cfg_1
        cfg.data = data_cfg_1
        cfg.train_platform = train_cfg.platform

    checkpoint_cfg = cfg.get("checkpoint_2", {})
    checkpoint = resolve_checkpoint(checkpoint_cfg)
    assert checkpoint is not None
    # load agent config and specs from checkpoint
    run, train_cfg, checkpoint_paths = checkpoint
    if isinstance(run, Path):
        run_name = run.name
    else:
        # wandb Api Run object
        run_name = run.id

    assert len(checkpoint_paths) == 1
    epoch, checkpoint_path = checkpoint_paths[0]
    metadata_2 = {
        "run_name": run_name,
        "epoch": epoch,
    }
    log.info(f"Loading checkpoint after epoch {epoch} of run {run_name}...")

    # merge the agent and data configs, with the current config taking precedence
    agent_cfg_2 = OmegaConf.merge(train_cfg.agent, agent_cfg)
    data_cfg_2 = merge_data_configs(train_cfg.data, data_cfg)

    agent_cfg_2 = patch_legacy_configs(agent_cfg_2)
    data_cfg_2 = patch_legacy_configs(data_cfg_2)

    # modify the _target_ to point to the module's load_from_checkpoint method
    agent_cfg_2 = patch_load_from_checkpoint(agent_cfg_2, checkpoint_path)

    update_wandb_config(cfg, train_tags, train_notes)

    # configure torch, e.g. set_float32_matmul_precision
    configure_torch(cfg.get("torch"))

    # instantiate dataset
    datamodule: TrajectoryDataModule = instantiate_datamodule(data_cfg_1)

    # manually run prepare data and setup so we can use dataset specs for model creation
    log.debug("Preparing and setting up data...")
    datamodule.prepare_data(stage="test")
    datamodule.setup(stage="test")

    # instantiate agent
    log.debug("Instantiating agents...")
    # recursively delete these fields in config dictionary
    # we want these to be saved to WandB but we don't want them for instantiation
    delete_keys_recursively(agent_cfg_1, ["name"])
    agent_1: BaseAgent = hydra.utils.instantiate(agent_cfg_1)
    agent_1.checkpoint_metadata = metadata_1

    delete_keys_recursively(agent_cfg_2, ["name"])
    agent_2: BaseAgent = hydra.utils.instantiate(agent_cfg_2)
    agent_2.checkpoint_metadata = metadata_2

    agents = [agent_1, agent_2]

    log.debug("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    callbacks.extend(datamodule.get_callbacks())

    log.debug("Instantiating trainer...")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, _target_=Trainer, callbacks=callbacks, logger=logger
    )

    log.info("Starting testing loop...")
    while True:
        try:
            for agent in agents:
                log.warning(f"Testing agent {agent.checkpoint_metadata['run_name']}...")
                trainer.test(agent, datamodule=datamodule)

        except KeyboardInterrupt:
            break

    log.info("Testing loop completed.")

    datamodule.close()


if __name__ == "__main__":
    setup_resolvers()
    test()
