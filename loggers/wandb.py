from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Sequence

import wandb
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers.wandb import WandbLogger as LightningWandbLogger
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


class WandbLogger(LightningWandbLogger):

    def __init__(
        self,
        *args,
        tags_from_overrides: bool = True,
        notes_from_overrides: bool = True,
        exclude_override_keys: Sequence[str] | None = None,
        save_dir: str | Path | None = None,
        slurm_specific_wandb_dirs: bool = True,
        **kwargs,
    ):
        exclude_override_keys = exclude_override_keys or []

        # enrich wandb notes and tags with hydra overrides
        overrides = HydraConfig.get().overrides.task

        # remove leading "+" and split into (key, value) pairs
        # (filter out values like ~debug that are not key-value pairs)
        overrides = [o.lstrip("+").split("=", 1) for o in overrides if "=" in o]
        overrides = {k: v for k, v in overrides}

        # ignore blank placeholder experiment "none"
        if "experiment" in overrides and overrides["experiment"] == "none":
            del overrides["experiment"]

        # ignore certain keys that almost always show up in the overrides
        # some of these will end up as tags instead
        param_overrides = {
            k: v for k, v in overrides.items() if k not in exclude_override_keys
        }

        if kwargs.get("notes") is None and notes_from_overrides and param_overrides:

            # shorten keys by taking only the last part of a dotted path
            # e.g. agent.obs_encoder.t10_image_tokenizer.fusion_type=cat -> t10_image_tokenizer.fusion_type=cat
            param_overrides = [
                ".".join(key.split(".")[-2:]) + "=" + value
                for key, value in param_overrides.items()
            ]
            param_overrides = ", ".join(param_overrides)

            kwargs["notes"] = f"Overrides: {param_overrides}"

        if tags_from_overrides:
            tags = kwargs.get("tags", []) or []
            if "obs_encoder" in overrides:
                tags.append(overrides["obs_encoder"])
            if "experiment" in overrides:
                # experiment may show up in the tags and the notes
                tags.append(overrides["experiment"])
            tags = tags or None  # avoid passing empty list to wandb
            kwargs["tags"] = tags

        if save_dir is not None:
            kwargs["dir"] = os.path.expandvars(save_dir)

        if slurm_specific_wandb_dirs:
            # create a unique wandb directory for each slurm task
            slurm_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
            slurm_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
            tmpdir = os.getenv("TMPDIR")
            if slurm_job_id is not None and slurm_task_id is not None:
                save_dir = os.path.join(
                    tmpdir or "wandb", f"{slurm_job_id}", f"{slurm_task_id}"
                )
                cache_dir = os.path.join(save_dir, "cache")
                os.environ["WANDB_DIR"] = save_dir
                os.environ["WANDB_CACHE_DIR"] = cache_dir
                log.info(
                    f"Using SLURM-specific wandb directory {save_dir} with cache dir {cache_dir}..."
                )
                kwargs["dir"] = save_dir

        super().__init__(*args, **kwargs)

        # Trigger lazy creation of wandb run by accessing experiment
        # We want to initialize the run as early as possible to log all the
        # messages from instantiating the dataset and model
        experiment = self.experiment
        if experiment is None:
            log.warning("Wandb run is not initialized, somehow...")
        else:
            log.info(
                f"Initialized wandb run with id {self.run_id} and name {self.experiment.name}..."
            )

        # Any test metrics logged with the "eval_metrics/" prefix will be associated
        # with the epoch of the checkpoint used for testing, not the wandb step
        # TODO: avoid hard-coding this
        wandb.run.define_metric("eval_metrics/*", step_metric="ckpt_epoch")
        wandb.run.define_metric("Episode_Termination/*", step_metric="ckpt_epoch")
        wandb.run.define_metric("Episode_Reward/*", step_metric="ckpt_epoch")
        wandb.run.define_metric("success", step_metric="ckpt_epoch", summary="max")


def update_wandb_config(
    cfg: DictConfig,
    extra_tags: Sequence[str] | None = None,
    default_notes: str | None = None,
) -> None:
    if wandb.run is not None:
        wandb_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)

        # remove notes from config before logging it to wandb, as it is not useful for sorting/filtering
        wandb_cfg.get("logger", {}).get("wandb", {}).pop("notes", None)  # type: ignore

        # don't remove tags from config, as this enables sorting/filtering by tags

        wandb.config.update(wandb_cfg, allow_val_change=True)

        # append extra tags, e.g. from the checkpoint's original run
        tags = wandb.run.tags or ()
        tags += tuple(extra_tags) if extra_tags else ()
        tags = tags or None  # # avoid passing empty tuple to wandb
        if tags:
            wandb.run.tags = tags

        if wandb.run.notes is None and default_notes is not None:
            wandb.run.notes = default_notes


# filenames are like epoch=0-step=25.ckpt
# this is the default checkpoint filename defined in lightning's ModelCheckpoint callback
CKPT_PATTERN = re.compile(r"^epoch=(\d+)-step=(\d+).ckpt$")


def resolve_checkpoint(
    cfg: DictConfig,
) -> tuple[Path | wandb.Run, DictConfig, list[tuple[int, Path]]] | None:
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
        # create a dict mapping epoch to model path
        try:
            ckpts_by_epoch = {
                int(CKPT_PATTERN.match(path.name).group(1)): path for path in ckpt_paths
            }
        except AttributeError:
            raise ValueError(
                f"Could not parse model paths in {log_dir / 'checkpoints'}. "
                "Make sure the filenames are in the format 'epoch=0-step=25.ckpt'."
            )

        ckpts_by_epoch = list(sorted(ckpts_by_epoch.items()))

        epochs = cfg.get("epochs", "last")

        if epochs == "last":
            ckpts_by_epoch = ckpts_by_epoch[-1:]
        elif epochs == "all":
            pass  # keep all checkpoints
        elif match := re.fullmatch(r"last_(\d+)", epochs):
            n = int(match.group(1))
            ckpts_by_epoch = ckpts_by_epoch[-n:]
        elif isinstance(epochs, (int, list)):
            if isinstance(epochs, int):
                epochs = [epochs]
            ckpts_by_epoch = [
                (epoch, path) for epoch, path in ckpts_by_epoch if epoch in epochs
            ]
            if len(ckpts_by_epoch) != len(epochs):
                found_epochs = [epoch for epoch, _ in ckpts_by_epoch]
                missing_epochs = set(epochs) - set(found_epochs)
                raise ValueError(
                    f"Could not find checkpoints for epochs {sorted(missing_epochs)} "
                    f"in folder {log_dir / 'checkpoints'}."
                )
        else:
            raise ValueError(
                f"Invalid value for cfg.epochs: {epochs}. "
                "Must be 'last', 'all', 'last_N' or a list of integers."
            )

        log.debug(f"Found {len(ckpts_by_epoch)} checkpoint(s) in folder {log_dir}...")

        return log_dir, train_cfg, ckpts_by_epoch

    run_id = cfg.get("wandb_run_id")
    if run_id is not None:
        # download model and specs artifacts from wandb, and get the config used
        # to train the model
        prefix = []
        if (project := cfg.get("wandb_project")) is not None:
            prefix.insert(0, project)
            if (entity := cfg.get("wandb_entity")) is not None:
                prefix.insert(0, entity)

        use_artifact = cfg.get("use_wandb_artifact", False)

        api = wandb.Api()
        run = api.run("/".join([entity, project, run_id]))

        # get config used to train the model
        train_cfg: DictConfig = OmegaConf.create(run.config)

        if (version := cfg.get("artifact_version")) is not None:
            # get a specific checkpoint version
            artifact_path = prefix + [f"model-{run_id}:{version}"]
            artifact_identifier = "/".join(artifact_path)

            log.debug(f"Loading checkpoint from wandb artifact {artifact_identifier}")

            if wandb.run is not None and not wandb.run.disabled and use_artifact:
                model_artifact = wandb.run.use_artifact(
                    artifact_identifier, type="model"
                )
            else:
                model_artifact = api.artifact(artifact_identifier, type="model")

            # download checkpoint from WandB
            ckpt_path = Path(model_artifact.download()) / "model.ckpt"

            # TODO: maybe get the epoch so that we always return int epochs as keys
            ckpt_paths = [(version, ckpt_path)]

        else:
            # get all artifacts from the run
            artifacts = [
                artifact
                for artifact in run.logged_artifacts()
                if artifact.type == "model"
            ]

            if not artifacts:
                raise RuntimeError(f"No artifacts found for wandb run {run_id}")

            try:
                if all(
                    CKPT_PATTERN.search(artifact.metadata.get("original_filename"))
                    for artifact in artifacts
                ):
                    # sort the artifacts by epoch, which can be extracted from the original filename
                    artifacts_by_epoch = {
                        int(
                            CKPT_PATTERN.match(
                                artifact.metadata["original_filename"]
                            ).group(1)
                        ): artifact
                        for artifact in artifacts
                    }
                else:
                    artifacts_by_epoch = {
                        int(artifact.history_step): artifact for artifact in artifacts
                    }

            except AttributeError:
                raise ValueError(
                    f"Could not parse artifact original filenames from run {run_id}. "
                    "Make sure the filenames are in the format 'epoch=0-step=25.ckpt'."
                )

            # sort dictionary entries by epoch
            artifacts_by_epoch = list(sorted(artifacts_by_epoch.items()))

            epochs = cfg.get("epochs", "last")

            try:
                _ = int(list(epochs)[0])  # type: ignore
                epochs = list(epochs)
            except ValueError:
                pass  # epochs is not iterable

            if isinstance(epochs, (int, list)):
                if isinstance(epochs, int):
                    epochs = [epochs]
                artifacts_by_epoch = [
                    (epoch, path)
                    for epoch, path in artifacts_by_epoch
                    if epoch in epochs
                ]
                if len(artifacts_by_epoch) != len(epochs):
                    found_epochs = [epoch for epoch, _ in artifacts_by_epoch]
                    missing_epochs = set(epochs) - set(found_epochs)
                    raise ValueError(
                        f"Could not find checkpoints for epochs {sorted(missing_epochs)} "
                        f"in run {run_id}."
                    )
            elif epochs == "last":
                artifacts_by_epoch = artifacts_by_epoch[-1:]
            elif epochs == "all":
                pass  # keep all artifacts
            elif match := re.fullmatch(r"last_(\d+)", epochs):
                n = int(match.group(1))
                artifacts_by_epoch = artifacts_by_epoch[-n:]
            elif match := re.fullmatch(r"last_(\d+)_step_(\d+)", epochs):
                n = int(match.group(1))
                step = int(match.group(2))
                # filter artifacts to only those matching the last N with given step between
                artifacts_by_epoch = artifacts_by_epoch[-1::-step][:n][::-1]
            elif match := re.fullmatch(r"spread_(\d+)", epochs):
                spread = int(match.group(1))
                indices = [
                    round((i + 1) * (len(artifacts_by_epoch) - 1) / (spread))
                    for i in range(spread)
                ]
                artifacts_by_epoch = [artifacts_by_epoch[i] for i in indices]
            else:
                raise ValueError(
                    f"Invalid value for cfg.epochs: {epochs}. "
                    "Must be 'last', 'all', 'last_N' or a list of integers."
                )

            # TODO: maybe if there is only one artifact, add support for use_artifact
            if wandb.run is not None and not wandb.run.disabled and use_artifact:
                for _, artifact in artifacts_by_epoch:
                    wandb.run.use_artifact(artifact, type="model")

            log.debug(
                f"Loading {len(artifacts_by_epoch)} checkpoint(s) from wandb run {run_id}..."
            )

            # download all selected checkpoints from WandB
            ckpt_paths = [
                (epoch, Path(artifact.download()) / "model.ckpt")
                for epoch, artifact in artifacts_by_epoch
            ]

        return run, train_cfg, ckpt_paths

    else:
        return None


def get_wandb_run_from_cfg(
    cfg: DictConfig, use_artifact: bool = False
) -> wandb.apis.public.Run:
    """
    Resolve the model artifact exactly similar to resolve_checkpoint and
    return the public Run object that logged the artifact.
    """
    run_id = cfg.get("wandb_run_id")
    if run_id is None:
        raise ValueError(
            "cfg.wandb_run_id is required to locate the run via the artifact."
        )

    version = cfg.get("artifact_version", "latest") or "latest"
    artifact_path = [f"model-{run_id}:{version}"]

    if (project := cfg.get("wandb_project")) is not None:
        artifact_path.insert(0, project)
        if (entity := cfg.get("wandb_entity")) is not None:
            artifact_path.insert(0, entity)

    artifact_identifier = "/".join(artifact_path)

    if wandb.run is not None and not wandb.run.disabled and use_artifact:
        model_artifact = wandb.run.use_artifact(artifact_identifier, type="model")
    else:
        api = wandb.Api()
        model_artifact = api.artifact(artifact_identifier, type="model")

    run = model_artifact.logged_by()
    if run is None:
        raise ValueError(
            f"Artifact {artifact_identifier} has no associated run (logged_by() returned None)."
        )
    return run


def upload_eval_videos(artifact_version: str, recording_dir: Path):
    if wandb.run is None or getattr(wandb.run, "disabled", False):
        raise RuntimeError(
            "update_wandb_table() must be called while a wandb run is active "
            "(e.g., from a Lightning callback like on_test_end)."
        )

    recording_paths = sorted(str(p.resolve()) for p in recording_dir.rglob("*.mp4"))

    # Log each video with an unique key
    # NOTE: Already uploaded videos will NOT be overwritten, new videos will be just appended
    for video_path in recording_paths:
        video = wandb.Video(
            video_path,
            format="mp4",
            caption=f"{artifact_version}",
        )
        wandb.log(
            {f"evaluation_artifact_version:{artifact_version}": video}, commit=True
        )
    # NOTE: Although WandbLogger finalizes wandb, still, if not calling explicitly finish, the videos are not uploaded everytime.
    wandb.finish()
