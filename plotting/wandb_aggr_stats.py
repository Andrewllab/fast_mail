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

    # Group runs by the key specified in cfg.group_by_config_keys and
    # by environment name (cfg.env_name_key)
    groups = defaultdict(lambda: defaultdict(list))
    for run in runs:
        # Verify that all runs have state finished and the expected number of epochs
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

        group_value = tuple(_get_nested(run.config, key) for key in group_by_keys)
        env_name = _get_nested(run.config, cfg.env_name_key)
        groups[group_value][env_name].append(run)

    logging.info(
        f"After filtering, {sum(len(runs) for envs in groups.values() for runs in envs.values())} runs remain."
    )

    # check if all groups have all expected environments
    for runs_by_env in groups.values():
        for expected_env in cfg.get("expected_envs", []):
            if expected_env not in runs_by_env:
                group_name = ", ".join(
                    f"{key}={value}" for key, value in zip(group_by_keys, group_value)
                )
                logging.warning(
                    f"Group {group_name} is missing expected environment {expected_env}."
                )

    group_stats = {}
    for group_value, runs_by_env in groups.items():

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

        all_eval_runs = [run for env_runs in runs_by_env.values() for run in env_runs]
        training_runs = list(
            set(
                _get_nested(run.config, "checkpoint.wandb_run_id")
                for run in all_eval_runs
            )
        )

        group_name = ", ".join(
            f"{key}={value}" for key, value in zip(group_by_keys, group_value)
        )
        logging.info(f"Group: {group_name}")
        logging.info(f"  Training run ids ({len(training_runs)} runs): {training_runs}")
        logging.info(f"  Number of eval runs: {len(all_eval_runs)}")
        for env, stats in env_stats.items():
            logging.info(
                f"  Success on {env}: {stats['mean']:.3f} ± {stats['std']:.3f}"
            )
        logging.info(f"  Overall: {group_mean:.3f} ± {group_std:.3f}")
        logging.info("")


if __name__ == "__main__":
    main()  # type: ignore[misc]
