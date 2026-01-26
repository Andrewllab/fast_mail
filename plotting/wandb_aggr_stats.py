from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Mapping

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

SUCCESS_X_AXIS = "ckpt_epoch"


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


@hydra.main(version_base=None, config_path="conf", config_name="wandb_aggr_stats")
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
    runs = list(api.runs(f"{cfg.entity}/{cfg.project}", filters=filters))
    logging.info(
        f"Found {len(runs)} runs with tag '{cfg.tags}' in project {cfg.entity}/{cfg.project}"
    )

    # Verify that all runs have state finished and the expected number of epochs
    for run in runs:
        if run.state != "finished":
            logging.warning(
                f"Run {run.id} (name={run.name}) is not finished (state={run.state}); "
                f"skipping this run."
            )
            continue

        if _get_nested(run.summary, "success.max") is None:
            logging.warning(
                f"Run {run.id} (name={run.name}) is missing 'success.max' in summary; "
                f"skipping this run."
            )
            continue

        if cfg.verify_n_ckpt_epochs:
            successes = run.history(
                keys=["success"], x_axis=SUCCESS_X_AXIS, pandas=False
            )
            if len(successes) != 3:
                logging.warning(
                    f"Run {run.id} (name={run.name}) has 'success' of length {len(successes)} != 3; "
                    f"skipping this run."
                )
                continue

    group_by_keys = cfg.group_by_config_keys
    if isinstance(group_by_keys, str):
        group_by_keys = [group_by_keys]

    # Group runs by the key specified in cfg.group_by_config_keys
    groups = defaultdict(list)
    for run in runs:
        group_value = tuple(_get_nested(run.config, key) for key in group_by_keys)
        groups[group_value].append(run)

    group_stats = {}
    for group_value, group_runs in groups.items():

        # group by environment
        runs_by_env = defaultdict(list)
        for run in group_runs:
            env_name = _get_nested(run.config, cfg.env_name_key)
            runs_by_env[env_name].append(run)

        expected_envs = cfg.get("expected_envs")
        for env in expected_envs:
            if env not in runs_by_env:
                logging.warning(
                    f"Group value {group_value} is missing expected environment {env}; "
                    f"skipping this group."
                )
                continue

        env_stats = {}
        for env, env_runs in runs_by_env.items():
            if len(env_runs) != cfg.get("expected_seeds", 3):
                logging.warning(
                    f"Group value {group_value} and environment {env} has "
                    f"{len(env_runs)} runs != expected {cfg.get('expected_seeds', 3)}; "
                    f"skipping this group."
                )
                continue

            successes = [_get_nested(run.summary, "success.max") for run in env_runs]

            if None in successes:
                logging.warning(
                    f"Group value {group_value} and environment {env} has missing success.max; "
                    f"skipping this group."
                )
                continue

            successes = np.array(successes)
            mean_success = np.mean(successes)
            std_success = np.std(successes, ddof=1)
            env_stats[env] = {"mean": mean_success, "std": std_success}

        n_envs = len(env_stats)
        group_mean = np.mean([v["mean"] for v in env_stats.values()])
        group_stds = np.array([v["std"] for v in env_stats.values()])
        group_std = np.sqrt((group_stds**2).sum()) / n_envs

        training_runs = list(
            set(
                _get_nested(run.config, "checkpoint.wandb_run_id") for run in group_runs
            )
        )

        group_name = ", ".join(
            f"{key}={value}" for key, value in zip(group_by_keys, group_value)
        )
        logging.info(f"Group: {group_name}")
        # logging.info(f"  Number of training runs: {len(training_runs)}")
        logging.info(f"  Training run ids ({len(training_runs)} runs): {training_runs}")
        logging.info(f"  Number of eval runs: {len(group_runs)}")
        logging.info(f"  SUCCESS: {group_mean:.3f} ± {group_std:.3f}")
        logging.info("")

        group_stats[group_value] = {
            "n_runs": len(group_runs),
            "n_envs": n_envs,
            "env_stats": env_stats,
            "group_mean": group_mean,
            "group_std": group_std,
        }


if __name__ == "__main__":
    main()  # type: ignore[misc]
