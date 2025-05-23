import logging
import os
from pathlib import Path
import hydra
import rootutils
from omegaconf import DictConfig

import subprocess
import psutil
import shutil

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

log = logging.getLogger(__name__)
project_root = os.path.dirname(os.path.abspath(__file__))


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
        p = subprocess.Popen(args=command_args, 
                            text=True,
                            bufsize=1,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            universal_newlines=True,
                            start_new_session=True
        ) 

        # the following try/except-block blocks and waits for the process's trigger or an error occurs 
        # we create a pipe to the process's output, using it as a trigger event based on the loggging outputs as IsaacSim's GUI gets stuck, so the process never actually finish even if we close the simulation application.
        try:
            for line in p.stdout:
                print(line, end="")
                if trigger_phrase in line:
                    print(f"[INFO] Subprocess finished ✅: trigger phrase detected — killing process tree...")
                    kill_process_tree(p.pid)

        except Exception as e:
            print(f"[ERROR] Error ⛔ while reading stdout: {e}")
            kill_process_tree(p.pid)

    except subprocess.CalledProcessError as e:
        print(f"Error while running test script: {e}")
        log_file.write(f"Error while running test script: {e}")


    log_file.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact_run_name", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--num_episodes", default=1)
    parser.add_argument("--eval_checkpoints", nargs='+', type=str)
    args = parser.parse_args()
    # api = wandb.Api()
    # run_name = cfg.artifact_run_name
    # project = cfg.get("artifact_project")
    # run = api.run(f"{project}/{run_name}")
    # all_artifacts = list(run.logged_artifacts())
    # specs_artifact = all_artifacts[0]
    # all_model_artifacts = all_artifacts[1:]


    save_root = ".live_test"
    save_root = Path(save_root)

    print(f"CONFIG: {args.experiment}")

    assert args.experiment is not None, "cfg.experiment is None. Cannot resolve defaults that depend on it."

    # Define all model versions and corresponding W&B run IDs
    for checkpoint_version in args.eval_checkpoints:
        print(f"Running evaluation for checkpoint {checkpoint_version}, run_id {args.artifact_run_name}")

        save_path = save_root / args.artifact_run_name / checkpoint_version
        save_path.mkdir(exist_ok=True, parents=True)
        log_path = save_path / "log.txt"


        # command = [
        #     "python", 
        #     "test.py",
        #     "-cn", 
        #     "test_isaac",
        #     f"artifact_version={checkpoint_version}",
        #     f"artifact_run_name={args.artifact_run_name}",
        #     f"experiment={args.experiment}",
        #     f"data.env_dataset.num_episodes={args.num_episodes}",
        #     "paths.recording_dir=recordings"
        # ],

        # Launch Isaac Sim through the AppLauncher simulation application
        # One could split the simulation start into a separate function and then call it, but making a separate function for one line is over-engineering.
        # app_launcher = AppLauncher()

        # TODO: trigger the kill_process_tree not by timeout, but by some other event!
        # Potential solutions: ID return status in env.close()?
        # try:
    

        # done_flag = Path("test_done.flag")
        # if done_flag.exists():
        #     done_flag.unlink()


        command_args=[
            "python", 
            "test.py",
            "-cn", 
            "test_isaac",
            f"artifact_version={checkpoint_version}",
            f"artifact_run_name={args.artifact_run_name}",
            f"experiment={args.experiment}",
            f"data.env_dataset.num_episodes={args.num_episodes}",
            "paths.recording_dir=recordings"
        ]
        trigger_phrase="Uploading evaluation videos to wandb completed"
        # p = subprocess.Popen(command, start_new_session=True)

        run_test_single_checkpoint(command_args, log_path=log_path, trigger_phrase=trigger_phrase)


        # while True:
        #     # Open the file in read mode
        #     file = open(log_path, "r")

        #     # Read each line one by one
        #     for line in file:
        #         if "done" in line:
        #             print("[INFO] Test completed, killing IsaacSim...")
        #             kill_process_tree(process.pid)
        #             break




        # if completed:
        #     print("[INFO] Test completed, killing IsaacSim...")
        #     kill_process_tree(process.pid)
        # p.wait(timeout=15)
        # print(f"Process ready just before killing it officially")


        # timeout_seconds = 300  # 5 min grace period
        # poll_interval = 5
        # elapsed = 0

        # while elapsed < timeout_seconds:
        #     if done_flag.exists():
                # print("[INFO] Test completed, killing IsaacSim...")
                # kill_process_tree(p.pid)
        #         return
        #     if p.poll() is not None:
        #         print("[INFO] Process exited early.")
        #         return
        #     time.sleep(poll_interval)
        #     elapsed += poll_interval


        # # except subprocess.TimeoutExpired:
        # kill_process_tree(p.pid)
        # except subprocess.CalledProcessError as e:
        #     print(f"Error while running test script: {e}")



        # cmd = ['/path/to/cmd', 'arg1', 'arg2']  # the external command to run
        # timeout_s = 10  # how many seconds to wait 

        # try:
        #     p = subprocess.Popen(cmd, start_new_session=True)
        #     p.wait(timeout=timeout_s)
        # except subprocess.TimeoutExpired:
        #     print(f'Timeout for {cmd} ({timeout_s}s) expired', file=sys.stderr)

        # print('Terminating the whole process group...', file=sys.stderr)
        # os.killpg(os.getpgid(p.pid), signal.SIGKILL) 

        # ps_output = p.stdout.read()
        # retcode = p.wait()
        # for pid_str in ps_output.strip().split("\n")[:-1]:
        #     os.kill(int(pid_str), signal.SIGTERM)

        # stdout = subprocess.PIPE if args.log_subprocess else None
        # stderr = subprocess.STDOUT if args.log_subprocess else None

        # process = subprocess.Popen(
        #     [
        #         "python", 
        #         "test.py",
        #         "-cn", 
        #         "test_isaac",
        #         f"artifact_version={checkpoint_version}",
        #         f"artifact_run_name={args.artifact_run_name}",
        #         f"experiment={args.experiment}",
        #         f"data.env_dataset.num_episodes={args.num_episodes}",
        #         "paths.recording_dir=recordings"
        #     ],
        #     cwd=project_root,
        #     text=True
        # )
        # try:
        #     output, _ = process.communicate()

        # # except subprocess.TimeoutExpired:
        # #     print(f"Timeout: Killing subprocess for checkpoint {checkpoint_version}")
        # #     process.kill()
        # #     process.wait()
        # except KeyboardInterrupt:
        #     print("Interrupted by user. Terminating subprocess...")
        #     process.kill()
        #     process.wait()
        #     break

        # except subprocess.CalledProcessError as e:
        #     print(f"Evaluation failed for {checkpoint_version}: {e}")


        # if process.returncode != 0:
        #     print(f"Evaluation failed for {checkpoint_version} with code {process.returncode}")
        # else:
        #     print(f"Evaluation completed for {checkpoint_version}")

    # remove all log files 
    shutil.rmtree(save_root)
    # remove all model checkpoints
    shutil.rmtree(f"artifacts/model-{args.artifact_run_name}:{checkpoint_version}")
    # remove all specs
    shutil.rmtree(f"artifacts/specs-{args.artifact_run_name}:v0")

if __name__ == "__main__":
    main()
