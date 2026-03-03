# Group 1: Tasks without handcam
# LiftPegUpright, PokeCube, PullCube, PushCube, PickCube, RollBall

import logging
import os
from pathlib import Path
from typing import Sequence

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
    EmbedSpec,
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
    ):
        self._root_dir = resolve_path(root_dir)
        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len

        self.files = list(iglob_follow_symlinks(self._root_dir, "**/*.h5"))

        if subfolders is not None:
            if isinstance(subfolders, str):
                subfolders = (subfolders,)

            log.info(
                f"Loading only data in the following subfolders: {list(subfolders)}"
            )

            all_files = []
            for subfolder in subfolders:
                files = [f for f in self.files if f.parent.name == subfolder]
                files = list(sorted(files, key=self.filename_keyfunc))
                files = get_subset(files, load_subset)
                all_files.extend(files)

            self.files = all_files

        else:
            self.files = list(sorted(self.files, key=self.filename_keyfunc))
            self.files = get_subset(self.files, load_subset)

        if not self.files:
            raise FileNotFoundError(
                f"No raw files found in {self._root_dir}. Please check the path."
            )

        self.trajs: list[tuple[Path, Group]] = [
            (file.relative_to(self._root_dir), h5py.File(str(file), "r"))
            for file in self.files
        ]

        traj_lengths = [len(traj["actions"]) for _, traj in self.trajs]

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

    def get_trajectory(self, traj_idx: int) -> TensorDict:
        path, traj = self.trajs[traj_idx]

        base_camera = traj["obs"]["sensor_data"]["base_camera"]
        rgb = base_camera["rgb"][:-1]
        # squeeze channel dimension and convert depth from float64 to float32 and from mm to m
        depth = base_camera["depth"][:-1, ..., 0].astype(np.float32) / 1000.0

        extrinsics = traj["obs"]["sensor_param"]["base_camera"]["cam2world_gl"][:-1]
        extrinsics = torch.from_numpy(extrinsics)
        extrinsics = convert_camera_frame_transform_convention(
            extrinsics, origin="opengl", target="ros"
        )

        action = traj["actions"][...]

        # verify that goal does not change over time
        goal_region = traj["env_states"]["actors"]["goal_region"]
        assert np.allclose(goal_region[...], goal_region[0], atol=1e-6)

        traj = TensorDict(
            {
                "obs": {
                    "base_camera": {"rgb": rgb, "depth": depth},
                    "robot_state": traj["obs"]["agent"]["qpos"][:-1],
                    "ee_pose": traj["obs"]["extra"]["tcp_pose"][:-1],
                    "base_cam_transform": extrinsics,
                    "goal_pos": goal_region[:-1, :3],
                },
                "action": action,
                "ref_action": action.copy(),
                "goal": {
                    "embed": traj["goal"]["preprocessed_embedding"][...],
                },
                # since each trajectory is stored in a separate file, we only need to store the path
                "path": str(path),
            },  # type: ignore
        )

        return traj

    def _load_specs(self) -> None:
        _, traj = self.trajs[0]

        obs_specs = {}
        T = len(traj["actions"])

        base_camera = traj["obs"]["sensor_data"]["base_camera"]

        rgb = base_camera["rgb"]
        match rgb.shape:
            case (t, height, width, 3) if t == T + 1:
                pass
            case _:
                raise ValueError(
                    f"Expected rgb stream from base_camera to have shape ({T+1}, H, W, 3), got {rgb.shape}"
                )

        depth = base_camera["depth"]
        match depth.shape:
            case (t, h, w, 1) if t == T + 1 and h == height and w == width:
                pass
            case _:
                raise ValueError(
                    f"Expected left camera depth images to have shape ({T+1}, {height}, {width}, 1), got {depth.shape}"
                )

        intrinsics = traj["obs"]["sensor_param"]["base_camera"]["intrinsic_cv"]
        assert intrinsics.shape == (T + 1, 3, 3)
        # verify that intrinsics do not change over time
        assert np.array_equiv(intrinsics[...], intrinsics[0])

        base_cam = CameraSpec(
            streams={
                "rgb": RGBStream(height, width, 3, channel_order="HWC"),
                "depth": DepthStream(height, width),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics[0], height=height, width=width
            ),
            extrinsics=torch.eye(4, dtype=torch.float32),
            dynamic_pose_obs_key="base_cam_transform",
        )
        obs_specs["base_camera"] = base_cam

        # robot state
        joint_pos = traj["obs"]["agent"]["qpos"]
        assert joint_pos.shape == (T + 1, 9)
        obs_specs["robot_state"] = ObsSpec(elem_shape=(9,), time=self.obs_seq_len)

        # goal region
        goal_pos = traj["env_states"]["actors"]["goal_region"]
        assert goal_pos.shape == (T + 1, 13)
        obs_specs["goal_pos"] = ObsSpec(elem_shape=(3,), time=self.obs_seq_len)

        # end-effector pose
        ee_pose = traj["obs"]["extra"]["tcp_pose"]
        assert ee_pose.shape == (T + 1, 7)
        # we concatenate ee_pos and ee_quat to get a shape of (7,)
        obs_specs["ee_pose"] = ObsSpec(elem_shape=(7,), time=self.obs_seq_len)
        obs_specs["target_ee_pose"] = ObsSpec(elem_shape=(7,), time=self.obs_seq_len)

        # base_cam_transform
        transform = traj["obs"]["sensor_param"]["base_camera"]["cam2world_gl"]
        assert transform.shape == (T + 1, 4, 4)
        obs_specs["base_cam_transform"] = ObsSpec(
            elem_shape=(4, 4), time=self.obs_seq_len
        )

        # actions
        actions = traj["actions"]
        assert actions.shape == (T, 8)
        action = ActionSpec(action_dim=8, time=self.action_seq_len)

        goal_embed = traj["goal"]["preprocessed_embedding"]
        assert goal_embed.shape == (1, 1024)
        goal = EmbedSpec(embed_dim=1024, n_tokens=1)

        self._specs = DataSpecs(
            obs=obs_specs,
            action=action,
            goal={"embed": goal},
        )
