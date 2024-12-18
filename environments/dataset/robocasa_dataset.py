import json
import os

import h5py
import torch

from environments.dataset.base_dataset import TrajectoryDataset


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
                print(f"Ignored short sequence #{i}: len={T}, window={self.window_size}")
            else:
                slices += [
                    (i, start, start + self.window_size) for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices
            
    def get_seq_length(self, idx):
        return self.demos[f"demo_{idx}"].attrs["num_samples"]
    
    def get_all_actions(self):
        result = []
        
        for demo in self.demos:
            result.append(torch.from_numpy(self.demos[demo]["actions"][:, :self.action_dim]))
            
        return torch.cat(result, dim=0).to(self.device)
    
    def get_all_observations(self):
        result = []
        
        for demo in self.demos:
            gripper_state = torch.from_numpy(self.demos[demo]["obs"]['robot0_joint_pos_cos'][:])
            joint_pos_sin = torch.from_numpy(self.demos[demo]["obs"]['robot0_joint_pos_sin'][:])
            joint_pos_cos = torch.from_numpy(self.demos[demo]["obs"]['robot0_joint_pos_cos'][:])
            
            robot_state = torch.cat([gripper_state, joint_pos_sin, joint_pos_cos], dim=1)
            result.append(robot_state)
        
        return torch.cat(result, dim=0).to(self.device)
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        
        demo = self.demos[f"demo_{i}"]
        
        action = torch.from_numpy(demo["actions"][start:end, :self.action_dim])
        
        obs = {}
        
        gripper_state = torch.from_numpy(demo["obs"]['robot0_gripper_qpos'][start:end])
        joint_pos_sin = torch.from_numpy(demo["obs"]['robot0_joint_pos_sin'][start:end])
        joint_pos_cos = torch.from_numpy(demo["obs"]['robot0_joint_pos_cos'][start:end])
        robot_state = torch.cat([gripper_state, joint_pos_sin, joint_pos_cos], dim=1)
        obs['robot_states'] = robot_state
        
        sampled_point_cloud = torch.from_numpy(demo["obs"]["sampled_point_cloud"][start:end])
        obs['sampled_point_cloud'] = sampled_point_cloud
        
        custom_sampled_point_cloud = torch.from_numpy(demo["obs"]["custom_sampled_point_cloud"][start:end])
        obs['custom_sampled_point_cloud'] = custom_sampled_point_cloud
        
        for cam_name in self.cam_names:
            rgb = torch.from_numpy(demo["obs"][f"{cam_name}_image"][start:end]).float().permute(0, 3, 1, 2) / 255.
            depth = torch.from_numpy(demo["obs"][f"{cam_name}_depth"][start:end]).float().permute(0, 3, 1, 2)
            
            obs[f"{cam_name}_image"] = rgb
            obs[f"{cam_name}_depth"] = depth
            
        obs["lang"] = json.loads(demo.attrs["ep_meta"])["lang"]
        
        return obs, action, torch.ones(action.shape[0]) # TODO is this mask correct?
