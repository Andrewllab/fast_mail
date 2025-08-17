from __future__ import annotations

import logging
import re
from pathlib import Path

import wandb
from lightning.pytorch.loggers.wandb import WandbLogger as LightningWandbLogger
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from typing_extensions import override
from wandb.wandb_run import Run

log = logging.getLogger(__name__)


class WandbLogger(LightningWandbLogger):
    @override
    @rank_zero_only
    def finalize(self, status: str) -> None:
        if status != "success":
            # If the run was aborted, we should finish it with an error
            # status to avoid any confusion in the UI.
            log.debug("Run was aborted, finishing wandb run with error status.")
            wandb.finish(exit_code=1)

        super().finalize(status)
        wandb.finish()


def init_wandb_logger(logger: LightningWandbLogger, cfg: DictConfig) -> None:
    wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # remove notes from config before logging it to wandb, as it is not useful for sorting/filtering
    wandb_cfg["logger"]["wandb"].pop("notes", None)  # type: ignore

    # don't remove tags from config, as this enables sorting/filtering by tags

    # save the rest of the config to wandb
    logger.experiment.config.update(wandb_cfg)

    # set root_dir in WandBLogger so it can access the hydra output folder
    # e.g. when saving hyperparameters
    # we can't do this before instantiating the logger, as this would also
    # set the wandb directory
    logger._save_dir = cfg.paths.output_dir


def resolve_checkpoint(
    cfg: DictConfig, use_artifact: bool = False
) -> tuple[DictConfig, Path] | None:
    """Get the model artifact from WandB and return the training config and model directory.
    Args:
        cfg: The configuration dictionary.
    Returns:
        A tuple containing the training config and the model directory.
    """
    log_dir = cfg.get("log_dir")
    if log_dir is not None:
        # retrieve training config and checkpoint path from local log_dir
        log_dir = Path(log_dir)

        cfg_path = log_dir / ".hydra/config.yaml"
        train_cfg: DictConfig = OmegaConf.load(cfg_path)

        ckpt_paths = (log_dir / "checkpoints").glob("*.ckpt")
        # filenames are like epoch=0-step=25.ckpt
        ckpt_pattern = re.compile(r"^epoch=(\d+)-step=(\d+)$")
        # create a dict mapping epoch to model path
        try:
            ckpt_paths = {
                int(ckpt_pattern.match(path.stem).group(1)): path for path in ckpt_paths
            }
        except AttributeError:
            raise ValueError(
                f"Could not parse model paths in {log_dir / 'checkpoints'}. "
                "Make sure the filenames are in the format 'epoch=0-step=25.ckpt'."
            )

        epoch = cfg.get("epoch")
        if epoch is None:
            # get the latest checkpoint
            epoch = max(ckpt_paths.keys())

        # get the checkpoint for the specified epoch
        try:
            ckpt_path = ckpt_paths[epoch]
        except KeyError:
            raise ValueError(
                f"Checkpoint for epoch {epoch} not found in {log_dir / 'checkpoints'}"
            )

        log.debug(f"Loading model from {ckpt_path} and config from {cfg_path}")

        return train_cfg, ckpt_path

    run_name = cfg.get("artifact_run_name")
    if run_name is not None:
        # download model and specs artifacts from wandb, and get the config used
        # to train the model
        version = cfg.get("artifact_version", "latest") or "latest"
        artifact_path = [f"model-{run_name}:{version}"]
        if (project := cfg.get("artifact_project")) is not None:
            artifact_path.insert(0, project)
            if (entity := cfg.get("artifact_entity")) is not None:
                artifact_path.insert(0, entity)

        artifact_identifier = "/".join(artifact_path)

        if wandb.run is not None and not wandb.run.disabled and use_artifact:
            model_artifact = wandb.run.use_artifact(artifact_identifier, type="model")
        else:
            api = wandb.Api()
            model_artifact = api.artifact(artifact_identifier, type="model")

        # get config used to train the model
        train_cfg = model_artifact.logged_by().config
        train_cfg: DictConfig = OmegaConf.create(train_cfg)

        # load saved model state
        ckpt_path = Path(model_artifact.download()) / "model.ckpt"

        return train_cfg, ckpt_path

    else:
        return None


def get_existing_table(artifact_run_name, key="evaluation"):
    api = wandb.Api()
    print(f"ARTIFACT RUN NAME: {artifact_run_name}")
    run = api.run(artifact_run_name)
    print(f"WHAT IS RUN ?????", run)
    for f in run.files():
        if f.name.startswith(key) and f.name.endswith(".table.json"):
            f.download(replace=True)
            return wandb.Table.read_json(f.name)
    return None


def update_wandb_table(cfg):
    # wandb.init(
    #     project=cfg.artifact_project,
    #     entity=cfg.artifact_entity,
    #     id=cfg.artifact_run_name,  # This is the original run ID from training
    #     resume="must"
    # )version
    checkpoint_version = cfg.get("artifact_version", "latest") or "latest"

    # recording_dir = Path(cfg.paths.recording_dir + version)
    recording_dir = Path(cfg.paths.recording_dir)
    recording_paths = [str(p.resolve()) for p in recording_dir.rglob("*.mp4")]

    # process all evaluation videos in a dict.
    episodes_by_ckpt = {f"{checkpoint_version}": []}
    for recording_path in recording_paths:
        episodes_by_ckpt[checkpoint_version].append(
            wandb.Video(recording_path, fps=30, format="mp4")
        )

    # Create evaluation table only the first time and get it for all subsequent model checkpoints
    table = get_existing_table(cfg.artifact_run_name)
    if table is None:
        table = wandb.Table(columns=["checkpoint_version", "video"])

    for version, video_list in sorted(episodes_by_ckpt.items()):
        table.add_data(version, video_list)

    wandb.log({"evaluation": table})
    wandb.finish()
