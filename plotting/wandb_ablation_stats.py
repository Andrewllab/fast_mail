import logging

import hydra
import numpy as np
import wandb
from omegaconf import DictConfig

EXPECTED_NUM_CHECKPOINTS = 3


@hydra.main(version_base=None, config_path="conf", config_name="wandb_ablation_stats")
def main(cfg: DictConfig) -> None:
    checkpoint_run_ids = cfg.checkpoint_run_ids
    expected_runs_per_checkpoint = cfg.expected_runs_per_checkpoint

    if not checkpoint_run_ids:
        logging.error("No checkpoint_run_ids provided in the config.")
        return

    api = wandb.Api()

    path = ""
    if (project := cfg.get("project")) is not None:
        path = project
        if (entity := cfg.get("entity")) is not None:
            path = f"{entity}/{project}"
    per_page = cfg.get("per_page")

    logging.info(f"Analyzing checkpoints for project: {path}")
    logging.info(f"Checkpoint run IDs: {checkpoint_run_ids}")

    checkpoint_means = []

    for checkpoint_id in checkpoint_run_ids:
        logging.info(f"\n=== Processing checkpoint {checkpoint_id} ===")

        # Query runs where config.checkpoint.wandb_run_id == checkpoint_id
        filters = {"config.checkpoint.wandb_run_id": checkpoint_id}

        # per_page is optional; if not set, wandb will use its default
        if per_page is not None:
            runs = api.runs(path, filters=filters, per_page=per_page)
        else:
            runs = api.runs(path, filters=filters)

        runs = list(runs)
        num_runs = len(runs)
        logging.info(f"Found {num_runs} runs for checkpoint {checkpoint_id}")

        # if num_runs != expected_runs_per_checkpoint:
        #     logging.warning(
        #         f"Expected {expected_runs_per_checkpoint} runs for checkpoint "
        #         f"{checkpoint_id}, but found {num_runs}. Skipping this checkpoint."
        #     )
        #     continue

        # Collect success.max for each run
        success_values = []
        for run in runs:
            if run.state != "finished":
                logging.warning(
                    f"Run {run.id} (name={run.name}) is not finished (state={run.state}); "
                    f"skipping this run."
                )
                continue

            success = run.summary.get("success")
            if success is not None:
                success = success.get("max")

            else:
                logging.warning(
                    f"Run {run.id} (name={run.name}) has no 'success.max' in summary; skipping this run."
                )
                continue

            if isinstance(success, (int, float)):
                success_values.append(success)
            else:
                logging.warning(
                    f"Run {run.id} (name={run.name}) has non-scalar 'success.max' "
                    f"({type(success)}); skipping this run."
                )

        if len(success_values) != expected_runs_per_checkpoint:
            logging.warning(
                f"Checkpoint {checkpoint_id}: only {len(success_values)} valid 'success.max' "
                f"values out of {expected_runs_per_checkpoint} runs. Skipping this checkpoint."
            )
            continue

        success_values_arr = np.array(success_values, dtype=np.float32)
        checkpoint_mean = success_values_arr.mean().item()
        checkpoint_std = success_values_arr.std(ddof=0).item()

        logging.info(
            f"Checkpoint {checkpoint_id}: mean success.max = {checkpoint_mean:.4f}, "
            f"std = {checkpoint_std:.4f}"
        )

        checkpoint_means.append(checkpoint_mean)

    # Aggregate across checkpoints
    if len(checkpoint_means) != EXPECTED_NUM_CHECKPOINTS:
        logging.error(
            f"Expected {EXPECTED_NUM_CHECKPOINTS} checkpoints with valid data but got {len(checkpoint_means)}."
        )
        return

    checkpoint_means_arr = np.array(checkpoint_means, dtype=np.float32)
    overall_mean = checkpoint_means_arr.mean().item()
    overall_std = checkpoint_means_arr.std(ddof=1).item()

    logging.info("\n=== Overall results ===")
    logging.info(f"Number of checkpoints included: {len(checkpoint_means)}")
    logging.info(f"Mean of per-checkpoint means: {overall_mean:.4f}")
    logging.info(f"Stddev of per-checkpoint means: {overall_std:.4f}")


if __name__ == "__main__":
    main()
