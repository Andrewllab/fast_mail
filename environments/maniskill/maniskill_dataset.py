import logging
import os
import re
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np
import torch
from h5py import Group
from tensordict import TensorDict

from environments.base_dataset import CustomHdf5Dataset, TrajectorySlices, get_subset
from environments.specs import (
    ActionSpec,
    CameraSpec,
    DataSpecs,
    DepthStream,
    ObsSpec,
    PinholeCameraIntrinsic,
    RGBStream,
)
from transforms.base_transform import TransformPartialsDict, init_transforms
from utils.math import convert_camera_frame_transform_convention
from utils.paths import iglob_follow_symlinks, resolve_path

log = logging.getLogger(__name__)


class ManiSkillDataset(CustomHdf5Dataset):
    def __init__(
        self,
        root_dir: os.PathLike,
        action_seq_len: int,
        obs_seq_len: int,
        item_transforms: TransformPartialsDict | None = None,
        load_subset: int | float | Sequence[int] | None = None,
        subfolders: Sequence[str] | None = None,
        steps_after_success: int = 0,
        rename_obs_keys: Mapping[str, str] | None = None,
    ):
        self._root_dir = resolve_path(root_dir)
        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len
        self.steps_after_success = steps_after_success

        if rename_obs_keys is None:
            self.rename_obs_keys = {
                "tcp_pose": "ee_pose",
                "qpos": "joint_pos",
                "qvel": "joint_vel",
            }
        else:
            self.rename_obs_keys = dict(rename_obs_keys)

        files = list(iglob_follow_symlinks(self._root_dir, "**/*.h5"))
        self.files = list(sorted(files, key=self.filename_keyfunc))

        if subfolders is not None:
            if isinstance(subfolders, str):
                subfolders = (subfolders,)
            subfolders_set = set(subfolders)
            files = [file for file in self.files if set(file.parts) & subfolders_set]
            log.info(
                f"Loading only data in the following subfolders: {list(subfolders)} ({len(files)} files out of {len(self.files)} total)"
            )
            self.files = files

        if not self.files:
            raise FileNotFoundError(
                f"No raw files found in {self._root_dir}. Please check the path."
            )

        self.trajs: list[tuple[Path, str, Group]] = []
        for file in self.files:
            h5_file = h5py.File(str(file), "r")

            traj_keys = list(sorted(h5_file.keys(), key=self.traj_name_keyfunc))
            traj_keys = get_subset(traj_keys, load_subset)

            self.trajs.extend(
                [
                    (file.relative_to(self._root_dir), traj_key, h5_file[traj_key])
                    for traj_key in traj_keys
                ]
            )

        if not self.trajs:
            raise ValueError(f"Subset {load_subset} resulted in zero trajectories.")

        traj_lengths = [
            np.argmax(traj["success"]) + 1 + self.steps_after_success
            for _, _, traj in self.trajs
        ]

        self.slices = TrajectorySlices(
            traj_lengths,
            obs_seq_len=obs_seq_len,
            action_seq_len=action_seq_len,
        )

        self._load_specs()

        if item_transforms is not None:
            log.debug("Instantiating item transforms...")
        self._item_transforms, self._specs = init_transforms(
            item_transforms, self._specs
        )

    @staticmethod
    def traj_name_keyfunc(key: str) -> int:
        # each trajectory is stored under a key like "traj_1", "traj_2", etc.
        match = re.fullmatch(r"traj_(\d+)", key)
        assert match is not None
        return int(match.group(1))

    def get_trajectory(self, traj_idx: int) -> TensorDict:
        path, name, traj = self.trajs[traj_idx]

        # actions
        action = traj["actions"][...]

        obs = {}

        # proprioception
        obs.update({key: value[...] for key, value in traj["obs"]["agent"].items()})

        # "extra" observations
        obs.update({key: value[...] for key, value in traj["obs"]["extra"].items()})

        # unsqueeze any scalar-valued observations to have shape (T, 1)
        obs = {
            key: value[:, np.newaxis] if value.ndim == 1 else value
            for key, value in obs.items()
        }

        # cameras
        for cam_name, cam_obs in traj["obs"]["sensor_data"].items():
            cam_param = traj["obs"]["sensor_param"][cam_name]
            obs[cam_name] = {}

            for stream_name, stream in cam_obs.items():
                if stream_name == "rgb":
                    stream = stream[...]
                elif stream_name == "depth":
                    # remove singleton channel, convert depth from int16 to
                    # float32 and convert from mm to m
                    stream = stream[..., 0].astype(np.float32) / 1000.0
                else:
                    raise ValueError(
                        f"Unknown stream name '{stream_name}' in camera '{cam_name}'"
                    )

                obs[cam_name][stream_name] = stream

            # camera extrinsics
            extrinsics = cam_param["cam2world_gl"][...]
            extrinsics = torch.from_numpy(extrinsics)
            extrinsics = convert_camera_frame_transform_convention(
                extrinsics, origin="opengl", target="ros"
            )
            obs[cam_name + "_transform"] = extrinsics

        # rename observation keys if needed
        for from_key, to_key in self.rename_obs_keys.items():
            if from_key in obs:
                obs[to_key] = obs.pop(from_key)

        td = TensorDict(
            {
                "obs": obs,
                "action": action,
                "ref_action": action.copy(),
                "path": str(path),
                "name": name,
            },  # type: ignore
        )
        # need to add a batch dimension for slicing to work
        td["obs"].auto_batch_size_(batch_dims=1)

        # trim trajectories to success + steps_after_success
        T = np.argmax(traj["success"]) + 1 + self.steps_after_success
        td["obs"] = td["obs"][:T]
        td["action"] = td["action"][:T]

        return td

    def _load_specs(self) -> None:
        _, _, traj = self.trajs[0]

        # actions
        actions = traj["actions"]
        match actions.shape:
            case (T, action_dim):
                pass
            case _:
                raise ValueError(
                    f"Expected actions to have shape (T, D), got {actions.shape}"
                )
        action = ActionSpec(action_dim=action_dim, time=self.action_seq_len)

        obs_specs = {}

        # proprioception
        for key, value in traj["obs"]["agent"].items():
            match value.shape:
                case (t, dim) if t == T + 1:
                    obs_specs[key] = ObsSpec(elem_shape=(dim,), time=self.obs_seq_len)
                case (t,) if t == T + 1:
                    obs_specs[key] = ObsSpec(elem_shape=(1,), time=self.obs_seq_len)
                case _:
                    raise ValueError(
                        f"Expected agent observation '{key}' to have shape ({T+1}, D), got {value.shape}"
                    )

        # "extra" observations
        for key, value in traj["obs"]["extra"].items():
            match value.shape:
                case (t, dim) if t == T + 1:
                    obs_specs[key] = ObsSpec(elem_shape=(dim,), time=self.obs_seq_len)
                case (t,) if t == T + 1:
                    obs_specs[key] = ObsSpec(elem_shape=(1,), time=self.obs_seq_len)
                case _:
                    raise ValueError(
                        f"Expected extra observation '{key}' to have shape ({T+1}, D), got {value.shape}"
                    )

        # cameras
        for cam_name, cam_obs in traj["obs"]["sensor_data"].items():
            cam_param = traj["obs"]["sensor_param"][cam_name]
            streams = {}

            if not len(list(cam_obs.items())):
                raise ValueError(f"Camera '{cam_name}' has no streams.")

            for stream_name, stream in cam_obs.items():
                if stream_name == "rgb":
                    match stream.shape:
                        case (t, height, width, 3) if t == T + 1:
                            pass
                        case _:
                            raise ValueError(
                                f"Expected rgb stream from camera '{cam_name}' to have shape ({T+1}, H, W, 3), got {stream.shape}"
                            )

                    streams[stream_name] = RGBStream(
                        height=height,
                        width=width,
                        channels=3,
                        channel_order="HWC",
                    )

                elif stream_name == "depth":
                    match stream.shape:
                        case (t, height, width, 1) if t == T + 1:
                            pass
                        case _:
                            raise ValueError(
                                f"Expected depth stream from camera '{cam_name}' to have shape ({T+1}, H, W, 1), got {stream.shape}"
                            )

                    streams[stream_name] = DepthStream(height=height, width=width)
                else:
                    raise ValueError(
                        f"Unknown stream name '{stream_name}' in camera '{cam_name}'"
                    )

            # intrinsics
            intrinsics = cam_param["intrinsic_cv"]
            assert intrinsics.shape == (T + 1, 3, 3)
            # verify that intrinsics do not change over time
            assert np.array_equiv(intrinsics[...], intrinsics[0])

            # extrinsics
            transform = cam_param["cam2world_gl"]
            assert transform.shape == (T + 1, 4, 4)
            obs_specs[cam_name + "_transform"] = ObsSpec(
                elem_shape=(4, 4), time=self.obs_seq_len
            )

            obs_specs[cam_name] = CameraSpec(
                streams=streams,
                time=self.obs_seq_len,
                intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                    intrinsics[0], height=height, width=width
                ),
                extrinsics=torch.eye(4, dtype=torch.float32),
                dynamic_pose_obs_key=cam_name + "_transform",
            )

        # rename observation keys if needed
        for from_key, to_key in self.rename_obs_keys.items():
            if from_key in obs_specs:
                obs_specs[to_key] = obs_specs.pop(from_key)

        self._specs = DataSpecs(
            obs=obs_specs,
            action=action,
        )
