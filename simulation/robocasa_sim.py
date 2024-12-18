import logging

import numpy as np
from omegaconf import ListConfig
import torch

from agents.base_agent import BaseAgent
from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS
from simulation.base_sim import BaseSim

log = logging.getLogger(__name__)


class RoboCasaSim(BaseSim):
    def __init__(
        self,
        env_name: str,
        camera_names: ListConfig[str],
        img_height: int,
        img_width: int,
        num_episode: int,
        max_step_per_episode: int,
        seed: int,
        device: str,
        render: bool = True,
        n_cores: int = 1,
        if_vision: bool = False,
    ):
        super().__init__(seed, device, render, n_cores, if_vision)

        self.num_episode = num_episode
        self.max_step_per_episode = max_step_per_episode
        self.camera_names = list(camera_names)

        self.env = REGISTERED_KITCHEN_ENVS[env_name](
            robots="PandaMobile",
            controller_configs={
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "kp": 150,
                "damping_ratio": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_ratio_limits": [0, 10],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "control_delta": True,
                "interpolation": None,
                "ramp_ratio": 0.2,
            },
            has_renderer=render,
            has_offscreen_renderer=True,
            camera_names=self.camera_names,
            use_camera_obs=True,
            camera_depths=True,
            camera_heights=img_height,
            camera_widths=img_width,
            # obj_registries=("objaverse", "aigen"),
            # camera_segmentations="geom",
            seed=seed
        )

        # self.env.seed(seed)

    def test_agent(self, agent: BaseAgent, cpu_set, epoch):
        success_count = 0
        task_completion_hold_count = -1

        for i in range(self.num_episode):
            obs = self.env.reset()

            if self.render:
                self.env.render()

            agent.reset()

            lang = self.env.get_ep_meta()["lang"]

            for j in range(self.max_step_per_episode):
                obs_dict = {}
                obs_dict["lang"] = lang

                gripper_state = torch.from_numpy(obs['robot0_gripper_qpos'])
                joint_pos_sin = torch.from_numpy(obs['robot0_joint_pos_sin'])
                joint_pos_cos = torch.from_numpy(obs['robot0_joint_pos_cos'])
                robot_state = torch.cat([gripper_state, joint_pos_sin, joint_pos_cos]).to(self.device).unsqueeze(0).unsqueeze(0)
                obs_dict["robot_states"] = robot_state

                # sampled_point_cloud = torch.from_numpy(obs["sampled_point_cloud"]).to(self.device)
                # obs_dict["sampled_point_cloud"] = sampled_point_cloud

                # custom_sampled_point_cloud = torch.from_numpy(obs["custom_sampled_point_cloud"]).to(self.device)
                # obs_dict["custom_sampled_point_cloud"] = custom_sampled_point_cloud

                for cam_name in self.camera_names:
                    rgb = torch.from_numpy(obs[f"{cam_name}_image"]).to(self.device).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0) / 255.
                    depth = torch.from_numpy(obs[f"{cam_name}_depth"]).to(self.device).float().permute(2, 0, 1).unsqueeze(0).unsqueeze(0)

                    obs_dict[f"{cam_name}_image"] = rgb
                    obs_dict[f"{cam_name}_depth"] = depth

                action = agent.predict(obs_dict).cpu().numpy()
                
                action = np.concatenate([action, np.array([0, 0, 0, 0, -1])])
                
                obs, _, done, _ = self.env.step(action)

                if self.render:
                    self.env.render()

                if self.env._check_success():
                    if task_completion_hold_count > 0:
                        task_completion_hold_count -= (
                            1  # latched state, decrement count
                        )
                    else:
                        task_completion_hold_count = (
                            10  # reset count on first success timestep
                        )
                else:
                    task_completion_hold_count = (
                        -1
                    )  # null the counter if there's no success

                if task_completion_hold_count == 0:
                    success_count += 1
                    done = True

                if done:
                    break

        success_rate = success_count / self.num_episode
        print(f"Success rate: {success_rate}")

        return success_rate
