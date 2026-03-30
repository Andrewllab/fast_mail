from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Mapping

import hydra
import numpy as np
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


def icm_and_ci(
    data: np.ndarray, percentile: float = 95
) -> tuple[np.ndarray, np.ndarray]:
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

    results_per_group = {}
    for group_value, runs_by_env in groups.items():

        # mapping from env name to list of dicts (one for each seed) mapping
        # checkpoint epoch to success rate
        success_rates: dict[str, list[dict[str, float]]] = {}
        for env, env_runs in runs_by_env.items():
            if len(env_runs) != cfg.get("expected_seeds", 3):
                logging.warning(
                    f"Group value {group_value} and environment {env} has "
                    f"{len(env_runs)} runs != expected {cfg.get('expected_seeds', 3)}; "
                    f"skipping this environment."
                )
                continue

            run_successes = []
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

                run_successes.append(
                    {
                        ckpt["ckpt_epoch"]: ckpt["success"]
                        for ckpt in success_per_checkpoint
                    }
                )

            success_rates[env] = run_successes

        results_per_env = {env: np.zeros(cfg.n_iterations) for env in success_rates}
        for env, success_by_checkpoint in success_rates.items():
            for i in range(cfg.n_iterations):

                # simulate a single experiment
                # 1. sample n_seeds seeds (or just take all seeds if n_seeds is None)
                # 2. sample n_rollouts rollouts for each checkpoint epoch and seed using a binomial distribution
                # 3. max across checkpoints
                # 4. mean across seeds

                if cfg.n_seeds is None:
                    seed_ids = range(len(success_by_checkpoint))
                else:
                    seed_ids = rng.choice(
                        len(success_by_checkpoint), size=cfg.n_seeds, replace=True
                    )

                results = []
                for seed in seed_ids:
                    result_by_ckpt = {}
                    for ckpt_epoch, p_success in success_by_checkpoint[seed].items():
                        result_by_ckpt[ckpt_epoch] = (
                            np.random.binomial(cfg.n_rollouts, p_success)
                            / cfg.n_rollouts
                        )
                    results.append(result_by_ckpt)

                max_results = [
                    max(result_by_ckpt.values()) for result_by_ckpt in results
                ]
                mean_result = np.array(max_results).mean()

                results_per_env[env][i] = mean_result

        task_suite_results = np.stack(list(results_per_env.values()), axis=-1)
        task_suite_results = task_suite_results.mean(axis=-1)
        results_per_env["overall"] = task_suite_results

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
        results_per_group[group_name] = {}

        logging.info(f"Group: {group_name}")
        logging.info(f"  Training run ids ({len(training_runs)} runs): {training_runs}")
        logging.info(f"  Number of eval runs: {len(all_eval_runs)}")
        for env, env_results in results_per_env.items():
            env_mean, env_bci = icm_and_ci(env_results)
            logging.info(f"  Success on {env}: {env_mean:.3f} ± {env_bci:.3f}")
            results_per_group[group_name][env] = (env_mean, env_bci)

        logging.info("")

    # save as yaml file
    yaml_data = {
        group_name: {
            env: {"mean": mean, "bci": bci} for env, (mean, bci) in env_results.items()
        }
        for group_name, env_results in results_per_group.items()
    }

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    with open(f"{output_dir}/results.yaml", "w") as f:
        OmegaConf.save(config=yaml_data, f=f.name)


if __name__ == "__main__":
    main()  # type: ignore[misc]
