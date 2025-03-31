import logging
import os
import pickle
from pathlib import Path

import torch
from tensordict import NonTensorData, TensorDict

from environments.datasets.base_dataset import TrajectoryDataset
from environments.specs import ActionSpec, CameraSpec, DataSpecs, Spec

log = logging.getLogger(__name__)


class LiberoDataset(TrajectoryDataset):
    def __init__(
        self,
        *args,
        embeddings_file: os.PathLike,
        trajs_per_task: int | float | None = None,
        **kwargs,
    ):
        self.embeddings_file = embeddings_file
        self.task_embeddings = None

        if "load_subset" in kwargs and trajs_per_task is None:
            trajs_per_task = kwargs.pop("load_subset")
        self.trajs_per_task = trajs_per_task

        super().__init__(*args, **kwargs)

    def find_raw_files(self) -> list[Path]:
        return list(self.root_dir.glob("*.hdf5"))

    def load_from_raw_file(self, filepath: Path) -> TensorDict | list[TensorDict]:
        log.debug(f"Loading trajectories from file {filepath}")

        if self.task_embeddings is None:
            with open(self.embeddings_file, "rb") as f:
                self.task_embeddings = pickle.load(f)

        # remove "_demo" from the end of the stem
        demo_name = filepath.stem[:-5]
        task_embedding = self.task_embeddings[demo_name]

        all_trajs = TensorDict.from_h5(str(filepath))

        # each trajectory is stored under a key like "demo_0", "demo_1", etc.
        demo_keys = list(
            sorted(all_trajs["data"].keys(), key=lambda demo_i: int(demo_i[5:]))
        )

        if isinstance(self.trajs_per_task, float):
            # if trajs_per_task is a fraction, take that fraction of the total
            # number of trajectories
            end = int(len(demo_keys) * self.trajs_per_task)
        else:
            end = self.trajs_per_task

        trajs = []
        for key in demo_keys[:end]:
            traj = all_trajs["data"][key]

            robot_state = torch.cat(
                (traj["obs", "joint_states"], traj["obs", "gripper_states"]), dim=-1
            )

            traj = TensorDict(
                {
                    "obs": {
                        "agentview": traj["obs", "agentview_rgb"],
                        "eye_in_hand": traj["obs", "eye_in_hand_rgb"],
                        "robot_state": robot_state.float(),
                    },
                    "action": traj["actions"].float(),
                },  # type: ignore
            )

            # wrap the goal in a NonTensorData so that the same goal is returned
            # for all time steps
            traj["goal"] = NonTensorData(TensorDict({"embed": task_embedding}))

            # set batch_size in TensorDict, otherwise it can't be indexed
            traj.auto_batch_size_()

            trajs.append(traj)

        return trajs

    def get_specs(self) -> DataSpecs:
        return DataSpecs(
            obs={
                "agentview": CameraSpec(
                    shape=(self.obs_seq_len, 128, 128, 3), type="rgb"
                ),
                "eye_in_hand": CameraSpec(
                    shape=(self.obs_seq_len, 128, 128, 3), type="rgb"
                ),
                "robot_state": Spec(shape=(self.obs_seq_len, 9), type="state"),
            },
            action=ActionSpec(shape=(self.action_seq_len, 7), type="action"),
            goal={
                "embed": Spec(shape=(1, 512), type="embed"),
            },
        )
