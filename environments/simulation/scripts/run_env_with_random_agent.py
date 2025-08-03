import argparse
import os
import sys
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# Parse CLI arguments
parser = argparse.ArgumentParser(
    description="Run random agent in IsaacLab environment."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to spawn."
)
parser.add_argument(
    "--task", type=str, default="Isaac-Insert-One-Leg-Franka-v0", help="Task name."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import all isaaclab related features after the simulation has started
from isaaclab_tasks.utils import parse_env_cfg

# TODO: remove the nasty hack of solving the ModuleNotFoundError
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import env registration module so envs are known to Gym
import furniture_bench


# Now start actual logic
def main():

    # parse configuration
    env_cfg = parse_env_cfg(
        task_name=args_cli.task,
        device=args_cli.device,
        num_envs=1,
    )

    # Create the IsaacLab environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="human", num_envs=args_cli.num_envs
    ).unwrapped

    # Reset env
    env.reset()
    count = 0

    print("[INFO]: Environment started with", args_cli.num_envs, "env(s).")

    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 300 == 0:
                env.reset()
                count = 0

            action = torch.randn_like(env.action_manager.action)
            obs, rew, terminated, truncated, info = env.step(action)

            print(f"[Step {count}] Reward: {rew[0].item():.3f}")
            count += 1

        env.sim.render()  # or rate_limiter.sleep() if you're throttling

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
