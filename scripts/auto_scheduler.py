import time
import subprocess
from pathlib import Path

import wandb
import yaml

POLL_INTERVAL = 600  # seconds
SUBMIT_TIMEOUT = 120  # seconds to wait for submitit to schedule the jobs
STATE_FILE = Path("scheduled_test_runs.yaml")


def load_scheduled_ids() -> set:
    if STATE_FILE.exists():
        data = yaml.safe_load(STATE_FILE.read_text())
        if data and "scheduled_run_ids" in data:
            return set(data["scheduled_run_ids"])
    return set()


def save_scheduled_ids(ids: set):
    STATE_FILE.write_text(yaml.dump({"scheduled_run_ids": sorted(ids)}, default_flow_style=False))


api = wandb.Api()

while True:
    scheduled_ids = load_scheduled_ids()

    runs = api.runs(
        "nicolasschreiber/PAKT_grid_search",
        filters={
            "$and": [
                {"state": "finished"},
                {"jobType": "train"},
                {"tags": {"$in": ["grid_search_06"]}},
            ]
        },
    )
    test_runs = api.runs(
        "nicolasschreiber/PAKT_grid_search",
        filters={
            "$and": [
                {"$or": [{"state": "finished"}, {"state": "running"}]},
                {"jobType": "test_sim"},
                {"tags": {"$in": ["grid_search_06"]}},
            ]
        },
    )

    new_ids = [r.id for r in runs if r.id not in scheduled_ids]

    for run in test_runs:
        try:
            run.load(force=True)  # Force a full reload from the API
            orig_train_run_id = run.config["checkpoint"]["wandb_run_id"]
            if orig_train_run_id in new_ids:
                new_ids.remove(orig_train_run_id)
        except Exception as e:
            pass


    if new_ids:
        ids_string = ",".join(new_ids)
        cmd = (
            f"MUJOCO_GL=egl HYDRA_FULL_ERROR=1 python test.py -cn test_robocasa"
            f" checkpoint.wandb_run_id={ids_string}"
            f" checkpoint.epochs=last"
            f" logger.wandb.project=PAKT_grid_search"
            f" checkpoint.wandb_project=PAKT_grid_search"
            f" platform=kluster"
            f" +agent.weights_only=False"
            f" agent.num_sampling_steps=10"
            f" data.env.num_episodes=50"
            f" data.env.record_video.disabled=False"
            f' +data.gpu_batch_transforms.t45_goal_embedding.goal_key="description"'
            f" --multirun"
        )
        print(f"Launching test for {len(new_ids)} new run(s): {ids_string}")
        print(cmd)
        # proc = subprocess.Popen(cmd, shell=True)

        # try:
        #     proc.wait(timeout=SUBMIT_TIMEOUT)
        #     print("Process exited on its own.")
        # except subprocess.TimeoutExpired:
        #     print(f"Killing process after {SUBMIT_TIMEOUT}s (submitit should have scheduled by now).")
        #     proc.kill()
        #     proc.wait()

        # scheduled_ids.update(new_ids)
        # save_scheduled_ids(scheduled_ids)
    else:
        print("No new finished runs. Sleeping...")

    time.sleep(POLL_INTERVAL)