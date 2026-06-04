from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import hydra
import numpy as np
import rootutils
import wandb
from omegaconf import DictConfig, OmegaConf

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from utils.trees import flatten_tree


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


def icm_and_ci(data: np.ndarray, percentile: float = 95) -> tuple[float, float]:
    percentile_lower = (100 - percentile) / 2
    percentile_upper = (100 + percentile) / 2
    q1, q3, ci_lower, ci_upper = np.percentile(
        data, [25, 75, percentile_lower, percentile_upper], axis=0
    )

    mask = (data >= q1) & (data <= q3)
    interquartile = data[mask]
    icm = np.mean(interquartile, axis=0)

    ci = max(ci_upper - icm, icm - ci_lower)

    return icm.item(), ci.item()


@hydra.main(version_base=None, config_path="conf", config_name="wandb_bci_stats")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    api = wandb.Api(timeout=cfg.get("wandb_timeout", 60))
    seed = cfg.get("seed")
    rng = np.random.default_rng(seed=seed)

    # Query runs with required tag
    required_tag = cfg.tags
    if isinstance(required_tag, str):
        required_tags = [required_tag]
    else:
        required_tags = list(required_tag)

    filters: dict[str, Any] = {
        "tags": {"$in": required_tags, "$nin": ["failed", "hidden"]},
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

    success_rates: dict[str, dict[str, dict[str, dict[int, float]]]] = {}
    for group_value, runs_by_env in groups.items():
        group_name = ", ".join(
            f"{key}={value}" for key, value in zip(group_by_keys, group_value)
        )

        success_rates[group_name] = {}

        # mapping from env name to list of dicts (one for each seed) mapping
        # checkpoint epoch to success rate
        for env, env_runs in runs_by_env.items():
            success_rates[group_name][env] = {}

            if len(env_runs) != cfg.get("expected_seeds", 3):
                logging.warning(
                    f"Group value {group_value} and environment {env} has "
                    f"{len(env_runs)} runs != expected {cfg.get('expected_seeds', 3)}; "
                    f"skipping this environment."
                )
                continue

            for run in env_runs:
                success_per_checkpoint = run.history(
                    keys=["success"], x_axis="ckpt_epoch", pandas=False
                )
                if len(success_per_checkpoint) != 3:
                    logging.warning(
                        f"Run {run.id} (name={run.name}) has 'success' of length {len(success_per_checkpoint)} != 3; "
                        f"skipping this run."
                    )
                    continue

                training_run_id = _get_nested(run.config, "checkpoint.wandb_run_id")

                success_rates[group_name][env][training_run_id] = {
                    ckpt["ckpt_epoch"]: ckpt["success"]
                    for ckpt in success_per_checkpoint
                }

    # save success rates pulled from wandb as yaml file
    with open(output_dir / "success_rates.yaml", "w") as f:
        OmegaConf.save(config=success_rates, f=f.name)

    bootstrap_results: dict[str, dict[str, np.ndarray]] = {}
    bci_stats: dict[str, dict[str, dict[str, float]]] = {}
    for group_name, success_by_env in success_rates.items():
        bootstrap_results[group_name] = {}
        bci_stats[group_name] = {}

        for env, success_by_run in success_by_env.items():
            bootstrap_results[group_name][env] = np.zeros(cfg.n_iterations)

            # drop run ids and convert to list of dicts mapping checkpoint epoch to success rate
            success_by_run = list(success_by_run.values())

            for i in range(cfg.n_iterations):

                # simulate a single experiment
                # 1. sample n_seeds seeds (or just take all seeds if n_seeds is None)
                # 2. sample n_rollouts rollouts for each checkpoint epoch and seed using a binomial distribution
                # 3. max across checkpoints
                # 4. mean across seeds

                if cfg.n_seeds is None:
                    seed_ids = range(len(success_by_run))
                else:
                    seed_ids = rng.choice(
                        len(success_by_run), size=cfg.n_seeds, replace=True
                    )

                results = []
                for seed in seed_ids:
                    result_by_ckpt = {}
                    for ckpt_epoch, p_success in success_by_run[seed].items():
                        result_by_ckpt[ckpt_epoch] = (
                            rng.binomial(cfg.n_rollouts, p_success) / cfg.n_rollouts
                        )
                    results.append(result_by_ckpt)

                max_results = [
                    max(result_by_ckpt.values()) for result_by_ckpt in results
                ]
                mean_result = np.array(max_results).mean()

                bootstrap_results[group_name][env][i] = mean_result

        task_suite_results = np.stack(
            list(bootstrap_results[group_name].values()), axis=-1
        )
        task_suite_results = task_suite_results.mean(axis=-1)
        bootstrap_results[group_name]["overall"] = task_suite_results

        for env, env_results in bootstrap_results[group_name].items():
            env_mean, env_bci = icm_and_ci(env_results)
            # logging.info(f"  Success on {env}: {env_mean:.3f} ± {env_bci:.3f}")
            bci_stats[group_name][env] = {"mean": env_mean, "bci": env_bci}

    # save bootstrap results as npz file
    np.savez(output_dir / "bootstrap_results.npz", **flatten_tree(bootstrap_results))

    # save aggregate bci stats as yaml file
    with open(output_dir / "bci_stats.yaml", "w") as f:
        OmegaConf.save(config=bci_stats, f=f.name)

    for group_name, stats_by_env in bci_stats.items():

        training_runs = list(
            set(
                training_run_id
                for success_by_run in success_rates[group_name].values()
                for training_run_id in success_by_run.keys()
            )
        )
        n_eval_runs = sum(
            len(success_by_run) for success_by_run in success_rates[group_name].values()
        )

        logging.info(f"Group: {group_name}")
        logging.info(f"  Training run ids ({len(training_runs)} runs): {training_runs}")
        logging.info(f"  Number of eval runs: {n_eval_runs}")
        for env, stats in stats_by_env.items():
            logging.info(
                f"  Success on {env}: {stats['mean']:.3f} ± {stats['bci']:.3f}"
            )

        logging.info("")


if __name__ == "__main__":
    main()  # type: ignore[misc]
