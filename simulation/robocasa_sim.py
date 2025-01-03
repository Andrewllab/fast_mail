import logging

import einops
import numpy as np
import torch
from omegaconf import ListConfig
from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS
from robosuite.utils.transform_utils import (
    axisangle2quat,
    mat2quat,
    quat2axisangle,
    quat2mat,
)

from agents.base_agent import BaseAgent
from environments.wrappers.point_cloud_sampling_wrapper import PointCloudSamplingWrapper
from environments.wrappers.point_cloud_wrapper import PointCloudWrapper
from environments.wrappers.robosuite_wrapper import RobosuiteWrapper
from simulation.base_sim import BaseSim
from utils.camera_utils import posRotMat2Mat, quat2Mat
from utils.point_cloud.sampling.fps_pc_sampler import FPSPointCloudSampler

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
        pc_num_points: int = 1024,
    ):
        super().__init__(seed, device, render, n_cores, if_vision)

        self.num_episode = num_episode
        self.max_step_per_episode = max_step_per_episode
        self.camera_names = list(camera_names)

        base_env = REGISTERED_KITCHEN_ENVS[env_name](
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
            seed=seed,
        )

        self.env = PointCloudSamplingWrapper(
            PointCloudWrapper(RobosuiteWrapper(base_env)),
            FPSPointCloudSampler(),
            pc_num_points,
        )

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

                gripper_state = torch.from_numpy(obs["robot0_gripper_qpos"]).float()
                gripper_state = einops.rearrange(gripper_state, "d -> 1 1 d").to(
                    self.device
                )

                eef_pos = torch.from_numpy(obs["robot0_eef_pos"]).float()
                eef_pos = einops.rearrange(eef_pos, "d -> 1 1 d").to(self.device)

                eef_rot = torch.from_numpy(quat2Mat(obs["robot0_eef_quat"])).float()
                dir1 = einops.rearrange(eef_rot[:, 0], "d -> 1 1 d").to(self.device)
                dir2 = einops.rearrange(eef_rot[:, 2], "d -> 1 1 d").to(self.device)

                gravity_dir = einops.rearrange(
                    torch.Tensor([0, 0, -1]).float(), "d -> 1 1 d"
                ).to(self.device)

                sampled_point_cloud = torch.from_numpy(
                    obs["sampled_point_cloud"][:, :3]
                ).float()
                sampled_point_cloud = einops.rearrange(
                    sampled_point_cloud, "num_points d -> 1 1 num_points d"
                ).to(self.device)

                obs_dict = {
                    "pc": sampled_point_cloud,
                    "robot_states": torch.cat(
                        [
                            eef_pos,
                            dir1,
                            dir2,
                            gravity_dir,
                            gripper_state[:, :, :1],
                        ],
                        dim=-1,
                    ),
                }

                global_action = agent.predict(obs_dict).cpu().numpy()
                global_action = np.concatenate(
                    [global_action[1:], global_action[:1], np.array([0, 0, 0, 0, -1])]
                )

                action = self.get_local_action(global_action)

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

    def get_local_action(self, global_action: np.ndarray) -> np.ndarray:
        base_mat = self.env.sim.data.get_site_xmat(
            f"mobilebase{self.env.robots[0].idn}_center"
        )

        global_action_pos = global_action[:3]
        global_action_axis_angle = global_action[3:6]
        global_action_mat = quat2mat(axisangle2quat(global_action_axis_angle))

        local_action_pos = np.linalg.inv(base_mat) @ global_action_pos
        local_action_mat = np.linalg.inv(base_mat) @ global_action_mat @ np.linalg.inv(base_mat).T
        local_action_axis_angle = quat2axisangle(mat2quat(local_action_mat))

        local_action = np.concatenate(
            [local_action_pos, local_action_axis_angle, global_action[6:]]
        )
        return local_action
