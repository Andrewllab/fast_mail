"""
Test multiple trained checkpoints against Isaac-Sim and upload evaluation videos to the
original Weights & Biases (W&B) training run. It orchestrates end-to-end evaluation for one or more artifact versions
(checkpoints) of a trained agent. For each requested checkpoint it:

1) Spawns a child Python process that runs `test.py -cn test_isaac` with the appropriate
   Hydra overrides (artifact version, recording dir, video prefix, wrappers, and W&B resume).
2) Streams the child process logs and watches for a *trigger phrase* emitted by the
   Lightning callback "upload_eval_videos_wandb" that uploads videos to W&B (e.g., "Uploading evaluation videos
   to wandb completed").
3) When the trigger appears, the script kills the entire Isaac-Sim process's tree to avoid GUI hangs and orphaned services.
4) Optionally cleans up the local recording directory created for this evaluation.

Why a subprocess + trigger and not merge the script logic into test.py directly?
1) Inspired from Nic's script
2) This allows us to reuse the entire test-script without changing it since uncertain how to do it in a modular way.
3) Isaac-Sim will leave long-running processes even after
the Python script finishes. Running the evaluation in a subprocess lets us stream logs, detect a trigger and and then terminate the full process group reliably.
Everything else I've tried didn't killed all IsaacSim subprocesses.

Inputs & important config:
- `artifact_run_name` (str): the W&B run ID from training; used to resume the same run so videos are attached to the original training run.
- `artifact_version` (List[str]): one or more checkpoint versions (e.g., "[v0, v1]").
- `paths.recording_dir` (Path): base folder under which we create "<recording_dir>/<artifact_run_name>/" for temporary video files.
- `data.env_dataset.num_episodes` (int): how many episodes to roll out per checkpoint.
- data.env_dataset.env.episode_length_s (int): length of each episode in seconds

The script expects that `test.py -cn test_isaac` has a Lightning callback that:
- uploads videos to the active W&B run, and
- logs the same trigger phrase printed here.

Example for testing the first two checkpoints, each for 2 episodes:
HYDRA_FULL_ERROR=1 python test_trained_agent.py -cn test_isaac artifact_run_name=9ay2vcs6 artifact_version=[v0,v1] data.env_dataset.num_episodes=2

Open questions:
1) Is there any better way to unite the logic with the test script while keeping the subprocess creation?
    1.1. One possible solution would be to not kill IsaacSim, but just replace the model checkpoints and only reset the environment?
2) Removing all evaluation videos and all model checkpoints ensures proper cleaning. Should we make them optional?
"""

import logging
import os
import signal
import subprocess
import sys
import time

import hydra
import psutil
import rootutils
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

log = logging.getLogger(__name__)


def kill_process_tree(pid):
    log.info(f"Killing process tree with root pid {pid}...")

    # this only works because we set start_new_session=True in Popen
    # therefore, the pid of the process is also the process group id
    os.killpg(pid, signal.SIGKILL)  # send the signal to all the process groups

    # try:
    #     parent = psutil.Process(pid)
    #     children = parent.children(recursive=True)
    #     for child in children:
    #         child.kill()
    #     parent.kill()
    # except psutil.NoSuchProcess:
    #     pass


def kill_process_tree_nicely(pid, timeout=5):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Try graceful first
    for proc in parent.children(recursive=True) + [parent]:
        try:
            proc.terminate()  # SIGTERM on Unix, TerminateProcess on Windows
        except psutil.NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(
        parent.children(recursive=True) + [parent], timeout=timeout
    )

    # Force kill stragglers
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


def run_test_single_checkpoint(command_args, trigger_phrase):
    log.info(f"Launching subprocess: {' '.join(command_args)}")

    try:
        # Popen is used, instead of "subprocess.run" because we want to read the process's output as well as get its id to kill it and all its child processes.
        p = subprocess.Popen(
            args=command_args,
            text=True,  # file objects stdin, stdout and stderr are opened in text mode
            bufsize=1,  # line-buffered (only usable if text=True)
            stderr=subprocess.STDOUT,  # merge stdout and stderr
            stdout=subprocess.PIPE,  # capture stdout (and stderr)
            # this means that the new subprocess is not a child of the current
            # process (different process group)
            # consequences:
            # subprocess does not receive SIGINT when we press Ctrl+C in the main process
            # os.killpg(p.pid, signal.SIGTERM) kills the entire process group of the child
            start_new_session=True,
        )

        # the following try/except-block blocks and waits for the process's trigger or an error occurs
        # we create a pipe to the process's output, using it as a trigger event based on the logging outputs as IsaacSim's GUI gets stuck, so the process never actually finish even if we close the simulation application.
        try:
            for line in p.stdout:
                print(line, end="")  # process stdout is sent to main process's stdout
                if trigger_phrase in line:
                    log.info(
                        "Subprocess finished: trigger phrase detected — killing process tree..."
                    )
                    break
                elif "[ERROR]" in line or "Traceback (most recent call last)" in line:
                    time.sleep(5.0)  # wait a bit for wandb sync to finish
                    log.error("Subprocess error detected — killing process tree...")
                    break

        except Exception as e:
            log.exception(f"[ERROR] Error while reading stdout: {e}")

    except subprocess.CalledProcessError as e:
        log.exception(f"Subprocess failed with error: {e}")
    finally:
        kill_process_tree(p.pid)


@hydra.main(version_base=None, config_path="configs", config_name="test_isaac")
def main(cfg: DictConfig):
    # we don't use the config at all. We just need hydra to recognize overrides
    # as if this was test.py -cn test_isaac

    command_args = [
        sys.executable,  # fully-qualified path to current python interpreter
        "test.py",
        "-cn",
        "test_isaac",
    ]
    # pass on any overrides from this job to the test script
    command_args += list(HydraConfig.get().overrides.task)

    # the trigger phase is required for the subprocess we create in run_test_single_checkpoint to stop and kill all isaac-sim child processes
    trigger_phrase = "Finished wandb run with exit_code=0"

    run_test_single_checkpoint(command_args, trigger_phrase=trigger_phrase)


if __name__ == "__main__":
    main()
