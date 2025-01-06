from copy import deepcopy
import h5py
import hydra
from omegaconf import DictConfig
from robocasa.utils.env_utils import create_env
from environments.dataset.robocasa_dataset import RobocasaDataset
from environments.wrappers.point_cloud_sampling_wrapper import PointCloudSamplingWrapper
from environments.wrappers.point_cloud_wrapper import PointCloudWrapper
from environments.wrappers.robosuite_wrapper import RobosuiteWrapper
from robosuite.utils.transform_utils import quat2mat, axisangle2quat, mat2quat, quat2axisangle
from tqdm import tqdm
import torch
import einops
import numpy as np

from utils.point_cloud.sampling.fps_pc_sampler import FPSPointCloudSampler


@hydra.main(
    config_path="configs", config_name="robocasa_config.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:

    dataset_path = "/home/i53/student/donat/master_thesis/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/processed_demo_2.hdf5"

    f = h5py.File(dataset_path, "r")

    env_name = "PnPCounterToCab"
    base_env = create_env(
        env_name=env_name,
        camera_widths=128,
        camera_heights=128,
        camera_names=["robot0_agentview_center"],
        render_onscreen=True,
    )

    device = "cuda"

    env = PointCloudSamplingWrapper(
        PointCloudWrapper(RobosuiteWrapper(base_env)),
        FPSPointCloudSampler(),
        1024,
    )

    ds = RobocasaDataset(
        cam_names=["robot0_agentview_center"],
        data_directory=dataset_path,
        device="cpu",
        action_dim=7,
        obs_dim=13,
        max_len_data=260,
        window_size=17,
    )
    i = 0

    agent = hydra.utils.instantiate(cfg.agents)
    agent.load_snapshot(
        "/home/i53/student/donat/fast_mail/logs/PnPCabToCounter/sweeps/equibot/2025-01-03/03-14-05/epoch_1100.pth"
    )

    for demo_name in f["data"]:
        demo = f["data"][demo_name]
        initial_state = dict(states=demo["states"][0])
        initial_state["model"] = demo.attrs["model_file"]
        initial_state["ep_meta"] = demo.attrs.get("ep_meta", None)
        obs = deepcopy(env.reset_to(initial_state))

        agent.reset()

        env.render()

        for j in tqdm(range(len(demo["actions"]))):
            # obs, action, mask = ds[i]

            obs_dict = {}

            gripper_state = torch.from_numpy(obs["robot0_gripper_qpos"][:1]).float()
            gripper_state = einops.rearrange(gripper_state, "d -> 1 1 d").to(device)

            eef_pos = torch.from_numpy(obs["robot0_eef_pos"]).float()
            eef_pos = einops.rearrange(eef_pos, "d -> 1 1 d").to(device)

            eef_quat = obs["robot0_eef_quat"]
            eef_rot = torch.from_numpy(quat2mat(eef_quat)).float()
            eef_rot = torch.cat([eef_rot[:, 0], eef_rot[:, 2]], dim=-1)
            eef_rot = einops.rearrange(eef_rot, "d -> 1 1 d").to(device)

            gravity_dir = einops.rearrange(
                torch.Tensor([0, 0, -1]).float(), "d -> 1 1 d"
            ).to(device)

            sampled_point_cloud = torch.from_numpy(obs["sampled_point_cloud"])[:, :3].float()
            sampled_point_cloud = einops.rearrange(
                sampled_point_cloud, "num_points d -> 1 1 num_points d"
            ).to(device)

            obs_dict = {
                "pc": sampled_point_cloud,
                "robot_states": torch.cat(
                    [
                        eef_pos,
                        eef_rot,
                        gravity_dir,
                        gripper_state[:, :, :1],
                    ],
                    dim=-1,
                ),
            }

            global_action = agent.predict(obs_dict).cpu().numpy()

            if j < 5:
                # global_action = np.concatenate([action[0, 1:].numpy(), action[0, :1].numpy(), np.array([0, 0, 0, 0, -1])])
                # local_action = get_local_action(env, global_action)
                local_action = np.array([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1])
                agent.rollout_step_counter = 0
            else:
                global_action = np.concatenate(
                    [global_action[1:], global_action[:1], np.array([0, 0, 0, 0, -1])]
                )
                local_action = get_local_action(env, global_action)

            
            obs, _, done, _ = env.step(local_action)
            env.render()
            
            i += 1
            
def get_local_action(env, global_action: np.ndarray) -> np.ndarray:
    base_mat = env.sim.data.get_site_xmat(
        f"mobilebase{env.robots[0].idn}_center"
    )

    global_action_pos = global_action[:3]
    global_action_axis_angle = global_action[3:6]
    global_action_mat = quat2mat(axisangle2quat(global_action_axis_angle))

    local_action_pos = base_mat.T @ global_action_pos
    local_action_mat = base_mat.T @ global_action_mat @ base_mat
    local_action_axis_angle = quat2axisangle(mat2quat(local_action_mat))

    local_action = np.concatenate(
        [local_action_pos, local_action_axis_angle, global_action[6:]]
    )
    return local_action



if __name__ == "__main__":
    main()
