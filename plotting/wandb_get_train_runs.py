from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Mapping

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf


def _get_nested(d: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """
    Retrieve a nested value from a (possibly nested) dict using dot-separated keys.
    """
    cur: Any = d
    for part in key.split("."):
        try:
            cur = cur[part]
        except (TypeError, KeyError):
            return default
    return cur


@hydra.main(version_base=None, config_path="conf", config_name="wandb_get_train_runs")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)

    api = wandb.Api(timeout=cfg.get("wandb_timeout", 60))

    # Query runs with required tag
    required_tag = cfg.tags
    if isinstance(required_tag, str):
        required_tags = [required_tag]
    else:
        required_tags = list(required_tag)

    filters: dict[str, Any] = {
        "tags": {"$in": required_tags},
        "jobType": {"$eq": cfg.job_type},
    }
    cfg_filters = cfg.get("cfg_filters", {}) or {}
    if cfg_filters:
        cfg_filters = OmegaConf.to_container(cfg_filters, resolve=True)
    filters.update({f"config.{key}": value for key, value in cfg_filters.items()})
    runs = list(api.runs(f"{cfg.entity}/{cfg.project}", filters=filters))
    logging.info(
        f"Found {len(runs)} runs with tag '{cfg.tags}' in project {cfg.entity}/{cfg.project}"
    )

    group_by_keys = cfg.group_by_config_keys
    if isinstance(group_by_keys, str):
        group_by_keys = [group_by_keys]

    groups = defaultdict(list)
    for run in runs:
        # Verify that all runs have state finished and the expected number of epochs
        if run.state != "finished":
            logging.warning(
                f"Run {run.id} (name={run.name}) is not finished (state={run.state}); "
                f"skipping this run."
            )
            continue

        group_value = tuple(_get_nested(run.config, key) for key in group_by_keys)
        groups[group_value].append(run)

    logging.info(
        f"After filtering, {sum(len(group_runs) for group_runs in groups.values())} runs remain."
    )

    for group_value, group_runs in groups.items():

        run_ids = ",".join([run.id for run in group_runs])

        group_name = ", ".join(
            f"{key}={value}" for key, value in zip(group_by_keys, group_value)
        )
        logging.info(f"Group: {group_name}")
        logging.info(f"  run IDs:  {run_ids}")
        logging.info("")


if __name__ == "__main__":
    main()  # type: ignore[misc]
