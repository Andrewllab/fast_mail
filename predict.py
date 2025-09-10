import logging
from pathlib import Path

import hydra
import rootutils
from lightning import Callback, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from agents.base_agent import BaseAgent
from environments.datamodule import TrajectoryDataModule
from loggers.wandb import resolve_checkpoint, update_wandb_config
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
from utils.logging import configure_logging, log_exception_and_finish_wandb
from utils.torch_conf import configure_torch

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs")
@log_exception_and_finish_wandb
def predict(cfg: DictConfig) -> None:
    # resolve the entire config to catch any errors early
    OmegaConf.resolve(cfg)

    configure_logging(cfg.python_logging)

    # init wandb first so we can log any info or errors from instantiating dataset and model
    log.debug("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    agent_cfg = cfg.get("agent", {})
    data_cfg = cfg.get("data", {})
    checkpoint_cfg = cfg.get("checkpoint", {})
    train_tags = None
    train_notes = None
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
        data_cfg = OmegaConf.merge(train_cfg.data, data_cfg)

        # modify the _target_ to point to the module's load_from_checkpoint method
        agent_cfg = patch_load_from_checkpoint(agent_cfg, checkpoint_path)

        # save the merged configs back to cfg so they can be logged to wandb
        cfg.agent = agent_cfg
        cfg.data = data_cfg

    update_wandb_config(cfg, train_tags, train_notes)

    # configure torch, e.g. set_float32_matmul_precision
    configure_torch(cfg.get("torch"))

    # instantiate dataset
    datamodule: TrajectoryDataModule = instantiate_datamodule(data_cfg)

    # manually run prepare data and setup so we can use dataset specs for model creation
    log.debug("Instantiating datamodule...")
    datamodule.prepare_data()
    datamodule.setup(stage="predict")

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
        # hparams are loaded from checkpoint
        datamodule_hparams = {}

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

    log.info("Starting prediction loop...")
    trainer.predict(agent, datamodule=datamodule)

    log.info("Prediction loop completed.")

    datamodule.close()


if __name__ == "__main__":
    setup_resolvers()
    predict()
