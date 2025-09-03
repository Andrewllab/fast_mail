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

import shutil
import subprocess
from pathlib import Path

import hydra
import psutil
import rootutils
from omegaconf import DictConfig, OmegaConf

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)


def kill_process_tree(pid):
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass


def run_test_single_checkpoint(command_args, log_path, trigger_phrase):
    print(f"[INFO] Launching subprocess: {' '.join(command_args)}")

    # create a file for logging
    log_file = open(log_path, "w")

    try:
        # Popen is used, instead of "subprocess.run" because we want to read the process's output as well as get its id to kill it and all its child processes.
        p = subprocess.Popen(
            args=command_args,
            text=True,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            start_new_session=True,
        )

        # the following try/except-block blocks and waits for the process's trigger or an error occurs
        # we create a pipe to the process's output, using it as a trigger event based on the logging outputs as IsaacSim's GUI gets stuck, so the process never actually finish even if we close the simulation application.
        try:
            for line in p.stdout:
                print(line, end="")
                if trigger_phrase in line:
                    print(
                        f"[INFO] Subprocess finished: trigger phrase detected — killing process tree..."
                    )
                    kill_process_tree(p.pid)

        except Exception as e:
            print(f"[ERROR] Error while reading stdout: {e}")
            kill_process_tree(p.pid)

    except subprocess.CalledProcessError as e:
        print(f"Error while running test script: {e}")
        log_file.write(f"Error while running test script: {e}")

    log_file.close()


@hydra.main(version_base=None, config_path="configs")
def main(cfg: DictConfig):
    # resolve the entire config to catch any errors early
    OmegaConf.resolve(cfg)

    record_dir = cfg.paths.get("recording_dir")
    # the wandb's model id
    artifact_run_name = cfg.get("artifact_run_name")
    # a list of all model checkpoints to be evaluated
    artifact_versions = cfg.get("artifact_version")
    # defines the number of episodes to record
    num_episodes = cfg.data.env_dataset.num_episodes
    episode_length = cfg.data.env_dataset.env.episode_length_s

    # run test script for each required artifact_version
    for artifact_version in artifact_versions:
        print(
            f"Running evaluation for run_id {artifact_run_name}, version {artifact_version}"
        )

        # create directory for storing log-data. The log.txt file is used to read the logging info from UploadEvalVideosToWandbOnTestEnd as a trigger_phrase for the subprocess to stop.
        artifact_specific_record_dir = Path(record_dir) / artifact_run_name
        artifact_specific_record_dir.mkdir(exist_ok=True, parents=True)
        log_path = artifact_specific_record_dir / "log.txt"

        # setup prefix for storing videos used by the VideoRecorder
        name_prefix = str("model-" + artifact_run_name + ":" + artifact_version)

        # TODO: rewrite this hard-coded command-list which might be automated by reading all values from the hydra config cfg and ovewriting only the desired.
        command_args = [
            "python",
            "test.py",
            "-cn",
            "test_isaac",
            f"artifact_version={artifact_version}",
            f"artifact_run_name={artifact_run_name}",
            f"data.env_dataset.num_episodes={num_episodes}",
            f"data.env_dataset.env.wrapper_cfgs.record_video.video_folder={artifact_specific_record_dir}",
            f"data.env_dataset.env.wrapper_cfgs.record_video.name_prefix={name_prefix}",
            f"data.env_dataset.env.episode_length_s={episode_length}",  # episode length in seconds
            "data.env_dataset.env.enabled_wrappers=[record_video,preprocess]",  # required to record episodes with the VideoRecorder
            f"logger.wandb.id={artifact_run_name}",  # required to attach evaluation videos to the original wandb training run instead of creating a new wandb run
            "logger.wandb.resume=must",  # required to attach evaluation videos to the original wandb training run instead of creating a new wandb run
        ]
        # the trigger phase is required for the subprocess we create in run_test_single_checkpoint to stop and kill all isaac-sim child processes
        trigger_phrase = "Uploading evaluation videos to wandb completed"

        run_test_single_checkpoint(
            command_args, log_path=log_path, trigger_phrase=trigger_phrase
        )

        # TODO: make optional
        # remove evalution videos and log files
        shutil.rmtree(artifact_specific_record_dir)
        # remove model checkpoint
        # shutil.rmtree(f"artifacts/model-{artifact_run_name}:{artifact_version}")


if __name__ == "__main__":
    main()
