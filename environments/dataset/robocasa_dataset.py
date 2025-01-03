import json
import os

import h5py
import torch

from environments.dataset.base_dataset import TrajectoryDataset
from robosuite.utils.transform_utils import quat2mat


class RobocasaDataset(TrajectoryDataset):
    def __init__(
        self,
        cam_names: list[str],
        data_directory: os.PathLike,
        device: str = "cpu",
        obs_dim: int = 20,
        action_dim: int = 7,
        max_len_data: int = 256,
        window_size: int = 1,
    ):
        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size,
        )

        self.cam_names = cam_names
        self.data_file = h5py.File(data_directory, "r")
        self.demos = self.data_file["data"]

        self.slices = self.get_slices()

    def get_slices(self):
        slices = []

        for demo in self.demos:
            i = int(demo.split("_")[1])
            T = self.get_seq_length(i)

            if T - self.window_size < 0:
                print(
                    f"Ignored short sequence #{i}: len={T}, window={self.window_size}"
                )
            else:
                slices += [
                    (i, start, start + self.window_size)
                    for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    def get_seq_length(self, idx):
        return self.demos[f"demo_{idx}"].attrs["num_samples"]

    def get_all_actions(self):
        assert False
        result = []

        for demo in self.demos:
            result.append(
                torch.from_numpy(self.demos[demo]["global_actions"][:, : self.action_dim])
            )

        return torch.cat(result, dim=0).to(self.device)

    def get_all_observations(self):
        assert False
        result = []

        for demo in self.demos:
            gripper_state = torch.from_numpy(
                self.demos[demo]["obs"]["robot0_joint_pos_cos"][:]
            )
            joint_pos_sin = torch.from_numpy(
                self.demos[demo]["obs"]["robot0_joint_pos_sin"][:]
            )
            joint_pos_cos = torch.from_numpy(
                self.demos[demo]["obs"]["robot0_joint_pos_cos"][:]
            )

            robot_state = torch.cat(
                [gripper_state, joint_pos_sin, joint_pos_cos], dim=1
            )
            result.append(robot_state)

        return torch.cat(result, dim=0).to(self.device)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]

        demo = self.demos[f"demo_{i}"]

        action = torch.from_numpy(
            demo["global_actions"][start:end, : self.action_dim]
        ).float()
        
        action = torch.cat([action[:, 6:], action[:, :6]], dim=-1)
        
        obs = {}

        gripper_state = torch.from_numpy(
            demo["obs"]["robot0_gripper_qpos"][start:end, :1]
        ).float()
        obs["gripper_state"] = gripper_state

        eef_pos = torch.from_numpy(demo["obs"]["robot0_eef_pos"][start:end]).float()
        obs["eef_pos"] = eef_pos

        eef_quat = demo["obs"]["robot0_eef_quat"][start:end]
        eef_rot_list = []
        for quat in eef_quat:
            eef_rot_list.append(torch.from_numpy(quat2mat(quat)).float())
        eef_rot = torch.stack(eef_rot_list)
        obs["eef_rot"] = torch.cat([eef_rot[:, :, 0], eef_rot[:, :, 2]], dim=-1)
        
        sampled_point_cloud = torch.from_numpy(
            demo["obs"]["custom_sampled_point_cloud"][start:end]
        ).float()
        obs["sampled_point_cloud"] = sampled_point_cloud

        obs["lang"] = json.loads(demo.attrs["ep_meta"])["lang"]

        return obs, action, torch.ones(action.shape[0])  # TODO is this mask correct?
