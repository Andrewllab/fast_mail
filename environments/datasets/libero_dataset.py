import logging
import os
import pickle

import h5py
import numpy as np
import torch
from tensordict import TensorDict

from environments.datasets.base_dataset import TrajectoryDataset
from environments.specs import ActionSpec, CameraSpec, DataSpecs, Spec
from transforms.base_transform import TransformPartialsDict, init_transforms

log = logging.getLogger(__name__)


class LiberoDataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory: os.PathLike,
        embeddings_directory: os.PathLike,
        task: str,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        max_len_data: int,
        window_size: int,
        transforms: TransformPartialsDict | None = None,
        start_idx: int = 0,
        traj_per_task: int = 1,
    ):
        self.data_directory = data_directory
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.max_len_data = max_len_data
        self.window_size = window_size

        self.data_dir = os.path.join(data_directory, task)
        log.info("Loading dataset from {}".format(self.data_dir))

        embeddings_file = os.path.join(embeddings_directory, f"{task}.pkl")
        with open(embeddings_file, "rb") as f:
            tasks = pickle.load(f)

        data_embs = []
        actions = []
        masks = []
        agentview_rgb = []
        eye_in_hand_rgb = []

        all_states = []

        file_list = os.listdir(self.data_dir)

        for file in file_list:
            if not file.endswith(".hdf5"):
                continue

            filename = os.path.basename(file).split(".")[0][:-5]
            task_emb = tasks[filename]

            f = h5py.File(os.path.join(self.data_dir, file), "r")

            log.info("Loading demo: {}".format(file))

            demo_keys_list = list(f["data"].keys())

            indices = np.argsort([int(elem[5:]) for elem in demo_keys_list])

            # load the states and actions in demos according to demo_keys_list
            for i in indices[start_idx : start_idx + traj_per_task]:

                demo_name = demo_keys_list[i]
                demo = f["data"][demo_name]
                demo_length = demo.attrs["num_samples"]

                # zero_states = np.zeros((1, self.max_len_data, self.state_dim), dtype=np.float32)
                zero_actions = np.zeros(
                    (1, self.max_len_data, self.action_dim), dtype=np.float32
                )
                # zero_rewards = np.zeros((1, self.max_len_data), dtype=np.float32)
                # zero_dones = np.zeros((1, self.max_len_data), dtype=np.float32)
                zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

                # states_data = demo['states'][:]
                action_data = demo["actions"][:]
                # rewards_data = demo['rewards'][:]
                # dones_data = demo['dones'][:]

                # zero_states[0, :demo_length, :] = states_data  # would be T0, ...,Tn-1, Tn, 0, 0
                zero_actions[0, :demo_length, :] = action_data
                # zero_rewards[0, :demo_length] = rewards_data
                # zero_dones[0, :demo_length] = dones_data
                zero_mask[0, :demo_length] = 1

                # the_last_state = states_data[-1][:]
                the_last_action = action_data[-1][:]
                # the_last_reward = rewards_data[-1]
                # the_last_done = dones_data[-1]

                # zero_agentview = np.zeros((self.max_len_data, H, W, C), dtype=np.float32)
                # zero_inhand = np.zeros((self.max_len_data, H, W, C), dtype=np.float32)
                agent_view = demo["obs"]["agentview_rgb"][:]
                eye_in_hand = demo["obs"]["eye_in_hand_rgb"][:]

                joint_states = demo["obs"]["joint_states"][:]
                gripper_states = demo["obs"]["gripper_states"][:]

                robot_states = np.concatenate((joint_states, gripper_states), axis=-1)

                # test_img = agent_view[0]
                # test_img = test_img[::-1, :, :]
                # test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
                # cv2.imshow("test_img", test_img)
                # cv2.waitKey(0)

                # states.append(zero_states)
                actions.append(zero_actions)
                # rewards.append(zero_rewards)
                # dones.append(zero_dones)
                masks.append(zero_mask)

                agentview_rgb.append(agent_view)
                eye_in_hand_rgb.append(eye_in_hand)

                all_states.append(robot_states)

                data_embs.append(task_emb)

            f.close()

        self.actions = torch.from_numpy(np.concatenate(actions))  # shape: B, T, D

        self.agentview_rgb = agentview_rgb
        self.eye_in_hand_rgb = eye_in_hand_rgb

        self.all_states = all_states

        self.data_embs = data_embs
        self.tasks = tasks

        self.masks = torch.from_numpy(np.concatenate(masks))

        self.num_data = len(self.agentview_rgb)

        self.slices = self.get_slices()

        all_actions = self.get_all_actions()

        self._base_specs = self._specs = DataSpecs(
            obs={
                "agentview_image": CameraSpec(shape=(1, 128, 128, 3), type="rgb"),
                "eye_in_hand_image": CameraSpec(shape=(1, 128, 128, 3), type="rgb"),
                "robot_state": Spec(shape=(1, 9), type="state"),
            },
            action=ActionSpec(
                shape=(10, 7),
                type="action",
                a_mean=all_actions.mean(0),
                a_std=all_actions.std(0),
                a_min=all_actions.min(0).values,
                a_max=all_actions.max(0).values,
            ),
            goal={
                "embed": Spec(shape=(1, 512), type="embed"),
            },
        )

        log.info(f"Action lower bounds across dataset:\n{self.specs.action.a_min}")
        log.info(f"Action upper bounds across dataset:\n{self.specs.action.a_max}")

        if transforms is not None:
            self.transform, self._specs = init_transforms(transforms, self._specs)
        else:
            self.transform = lambda x: x

    @property
    def specs(self):
        return self._specs

    def get_slices(self):  # Extract sample slices that meet certain conditions
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

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
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        """
        Returns all actions from all trajectories, concatenated on dim 0 (time).
        """
        result = []
        # mask out invalid observations
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.agentview_rgb[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx: int) -> TensorDict:

        i, start, end = self.slices[idx]

        obs = {}

        task_emb = self.data_embs[i]

        agentview_rgb = self.agentview_rgb[i][start : start + 1]
        eye_in_hand_rgb = self.eye_in_hand_rgb[i][start : start + 1]

        robot_states = self.all_states[i][start : start + 1]

        agentview_rgb = torch.from_numpy(agentview_rgb)
        eye_in_hand_rgb = torch.from_numpy(eye_in_hand_rgb)

        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        obs["agentview_image"] = agentview_rgb
        obs["eye_in_hand_image"] = eye_in_hand_rgb

        obs["robot_state"] = torch.from_numpy(robot_states).float()

        item = TensorDict(
            {"obs": obs, "action": act, "goal": {"embed": task_emb}, "mask": mask},
            batch_size=(),
            device="cpu",
        )

        item = self.transform(item)

        return item
