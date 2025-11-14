import logging
import os
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import torch
from h5py import Group
from tensordict import TensorDict

from environments.base_dataset import (
    CustomHdf5Dataset,
    TrajectorySlices,
    get_subset,
    iglob_follow_symlinks,
)
from environments.specs import (
    ActionSpec,
    CameraSpec,
    DataSpecs,
    DepthStream,
    ImageStream,
    ObsSpec,
    PinholeCameraIntrinsic,
    RGBStream,
)
from transforms.base_transform import TransformPartialsDict, init_transforms
from utils.math import convert_quat, make_pose, quaternion_to_matrix
from utils.paths import resolve_path

log = logging.getLogger(__name__)


class RealRobotDataset(CustomHdf5Dataset):
    def __init__(
        self,
        root_dir: os.PathLike,
        action_seq_len: int,
        obs_seq_len: int,
        item_transforms: TransformPartialsDict | None = None,
        load_subset: int | float | Sequence[int] | None = None,
        subfolders: Sequence[str] | None = None,
        extrinsics: Mapping[str, Mapping[str, list[list[float]]]] | None = None,
    ):
        self._root_dir = resolve_path(root_dir)
        self.action_seq_len = action_seq_len
        self.obs_seq_len = obs_seq_len
        self._extrinsics = extrinsics

        # Recursively search for h5 files
        # Data collector saves files with datetime pattern: YYYY_MM_DD-HH_MM_SS.h5
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

        self.files = get_subset(self.files, load_subset)

        self.trajs: list[tuple[Path, str, Group]] = [
            (file.relative_to(self._root_dir), file.stem, h5py.File(str(file), "r"))
            for file in self.files
        ]

        traj_lengths = [len(traj["action"]["joint_pos"]) for _, _, traj in self.trajs]

        self.slices = TrajectorySlices(
            traj_lengths,
            obs_seq_len=obs_seq_len,
            action_seq_len=action_seq_len,
        )

        self._load_specs()

        if item_transforms is not None:
            log.debug("Instantiating item transforms...")
        self.transform, self._specs = init_transforms(item_transforms, self._specs)

    def get_trajectory(self, traj_idx: int) -> TensorDict:
        path, name, traj = self.trajs[traj_idx]

        # shape: (T, 7), float32
        joint_pos = traj["obs"]["proprioception"]["joint_pos"][...]
        # shape: (T, 1), float32
        gripper_pos = traj["obs"]["proprioception"]["gripper_pos"][...][:, None]
        # shape: (T, 3), float32
        ee_pos = traj["obs"]["proprioception"]["eef_pos"][...]
        # shape: (T, 4), float32
        ee_quat = traj["obs"]["proprioception"]["eef_quat"][...]
        # shape: (T, 7), float32
        target_joint_pos = traj["action"]["joint_pos"][...]
        # shape: (T, 1), float32
        target_gripper_pos = traj["action"]["gripper_pos"][...][:, None]
        # shape: (T, 4), float32
        target_ee_pos = traj["action"]["eef_pos"][...]
        # shape: (T, 3), float32
        target_ee_quat = traj["action"]["eef_quat"][...]

        robot_state = torch.cat(
            (torch.from_numpy(joint_pos), torch.from_numpy(gripper_pos)),
            dim=-1,
        )

        ee_pos = torch.from_numpy(ee_pos)
        ee_quat = convert_quat(torch.from_numpy(ee_quat), to="wxyz")
        ee_pose = torch.cat((ee_pos, ee_quat), dim=-1)

        ee_rot = quaternion_to_matrix(ee_quat)
        ee_transform = make_pose(ee_pos, ee_rot)
        assert torch.allclose(
            torch.from_numpy(
                traj["obs"]["gripper_cam"]["frames"]["dynamic_extrinsics"][...]
            ),
            ee_transform,
            atol=1e-6,
        )

        target_ee_pos = torch.from_numpy(target_ee_pos)
        target_ee_quat = convert_quat(torch.from_numpy(target_ee_quat), to="wxyz")
        target_ee_pose = torch.cat((target_ee_pos, target_ee_quat), dim=-1)

        action = torch.cat(
            (
                torch.from_numpy(target_joint_pos),
                torch.from_numpy(target_gripper_pos),
            ),
            dim=-1,
        )

        td = TensorDict(
            {
                "obs": {
                    "front_left_cam": {
                        "depth": traj["obs"]["left_cam"]["frames"]["depth"][...],
                        "left": traj["obs"]["left_cam"]["frames"]["left"][...],
                        "right": traj["obs"]["left_cam"]["frames"]["right"][...],
                    },
                    "front_right_cam": {
                        "depth": traj["obs"]["right_cam"]["frames"]["depth"][...],
                        "left": traj["obs"]["right_cam"]["frames"]["left"][...],
                        "right": traj["obs"]["right_cam"]["frames"]["right"][...],
                    },
                    "gripper_cam": {
                        "rgb": traj["obs"]["gripper_cam"]["frames"]["rgb"][...],
                        "depth": traj["obs"]["gripper_cam"]["frames"]["depth"][...],
                        "left": traj["obs"]["gripper_cam"]["frames"]["left"][...],
                        "right": traj["obs"]["gripper_cam"]["frames"]["right"][...],
                    },
                    "robot_state": robot_state,
                    "ee_pose": ee_pose,
                    "target_ee_pose": target_ee_pose,
                    "target_joint_pos": target_joint_pos,
                    "target_gripper_pos": target_gripper_pos,
                    "ee_transform": ee_transform,
                },
                "action": action,
                "ref_action": action.clone(),
                "path": str(path),
                "name": name,
            }  # type: ignore
        )

        # add a batch dimension so we can index
        td["obs"].auto_batch_size_(batch_dims=1)

        return td

    def _load_specs(self) -> None:
        _, _, traj = self.trajs[0]

        obs_specs = {}
        T = traj["obs"]["left_cam"]["frames"]["left"].shape[0]

        # static Zed Mini cameras
        for cam in ["left_cam", "right_cam"]:
            for stream in ["left", "right"]:
                rgb = traj["obs"][cam]["frames"][stream]

                match rgb.shape:
                    case (t, height, width, 3) if t == T:
                        pass
                    case _:
                        raise ValueError(
                            f"Expected {stream} stream from {cam} to have shape ({T}, H, W, 3), got {rgb.shape}"
                        )

            depth = traj["obs"][cam]["frames"]["depth"]
            match depth.shape:
                case (t, h, w) if t == T and h == height and w == width:
                    pass
                case _:
                    raise ValueError(
                        f"Expected depth stream from {cam} to have shape ({T}, {height}, {width}), got {depth.shape}"
                    )

            intrinsics = traj["obs"][cam]["meta"]["intrinsics"][...]
            assert intrinsics.shape == (3, 3)

            baseline = traj["obs"][cam]["meta"]["baseline"][...].item()

            if self._extrinsics is not None:
                extrinsics = self._extrinsics[f"front_{cam}"]["extrinsics"]
                extrinsics = torch.as_tensor(extrinsics, dtype=torch.float32)
                assert extrinsics.shape == (4, 4)
            else:
                extrinsics = None

            cam_spec = CameraSpec(
                streams={
                    "left": RGBStream(height, width, 3, channel_order="HWC"),
                    "right": RGBStream(height, width, 3, channel_order="HWC"),
                    "depth": DepthStream(height, width),
                },
                time=self.obs_seq_len,
                intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                    intrinsics, height=height, width=width
                ),
                extrinsics=extrinsics,
                baseline=baseline,
            )
            obs_specs[f"front_{cam}"] = cam_spec

        # Realsense D405 gripper camera
        rgb = traj["obs"]["gripper_cam"]["frames"]["rgb"]
        match rgb.shape:
            case (t, height, width, 3) if t == T:
                pass
            case _:
                raise ValueError(
                    f"Expected {stream} stream from gripper_cam to have shape ({T}, H, W, 3), got {rgb.shape}"
                )

        for stream in ["left", "right"]:
            ir = traj["obs"]["gripper_cam"]["frames"][stream]

            match ir.shape:
                case (t, h, w) if t == T and h == height and w == width:
                    pass
                case _:
                    raise ValueError(
                        f"Expected {stream} stream from gripper_cam to have shape ({T}, {height}, {width}), got {ir.shape}"
                    )

        depth = traj["obs"]["gripper_cam"]["frames"]["depth"]
        match depth.shape:
            case (t, h, w) if t == T and h == height and w == width:
                pass
            case _:
                raise ValueError(
                    f"Expected depth stream from gripper_cam to have shape ({T}, {height}, {width}), got {depth.shape}"
                )

        intrinsics = traj["obs"]["gripper_cam"]["meta"]["intrinsics"][...]
        assert intrinsics.shape == (3, 3)

        baseline = traj["obs"]["gripper_cam"]["meta"]["baseline"][...].item()

        if self._extrinsics is not None:
            extrinsics = self._extrinsics["gripper_cam"]["extrinsics"]
            extrinsics = torch.as_tensor(extrinsics, dtype=torch.float32)
            assert extrinsics.shape == (4, 4)
        else:
            extrinsics = None

        cam_spec = CameraSpec(
            streams={
                "rgb": RGBStream(height, width, 3, channel_order="HWC"),
                "depth": DepthStream(height, width),
                "left": ImageStream(height, width, channel_order="HW"),
                "right": ImageStream(height, width, channel_order="HW"),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=height, width=width
            ),
            extrinsics=extrinsics,
            dynamic_pose_obs_key="ee_transform",
            baseline=baseline,
        )
        obs_specs["gripper_cam"] = cam_spec

        # proprioception
        # robot_state
        joint_pos = traj["obs"]["proprioception"]["joint_pos"]
        assert joint_pos.shape == (T, 7)
        gripper_pos = traj["obs"]["proprioception"]["gripper_pos"]
        assert gripper_pos.shape == (T,)
        # we concatenate joint_pos and gripper_pos to get a shape of (T, 8)
        obs_specs["robot_state"] = ObsSpec(elem_shape=(8,), time=self.obs_seq_len)

        # ee_pose
        ee_pos = traj["obs"]["proprioception"]["eef_pos"]
        ee_quat = traj["obs"]["proprioception"]["eef_quat"]
        assert ee_pos.shape == (T, 3)
        assert ee_quat.shape == (T, 4)
        # we concatenate ee_pos and ee_quat to get a shape of (T, 7)
        obs_specs["ee_pose"] = ObsSpec(elem_shape=(7,), time=self.obs_seq_len)

        # target_joint_pos
        target_joint_pos = traj["action"]["joint_pos"]
        assert target_joint_pos.shape == (T, 7)
        obs_specs["target_joint_pos"] = ObsSpec(elem_shape=(7,), time=self.obs_seq_len)

        # target_gripper_pos
        target_gripper_pos = traj["action"]["gripper_pos"]
        assert target_gripper_pos.shape == (T,)
        obs_specs["target_gripper_pos"] = ObsSpec(
            elem_shape=(1,), time=self.obs_seq_len
        )

        # target_ee_pose
        ee_pos = traj["action"]["eef_pos"]
        ee_quat = traj["action"]["eef_quat"]
        assert ee_pos.shape == (T, 3)
        assert ee_quat.shape == (T, 4)
        obs_specs["target_ee_pose"] = ObsSpec(elem_shape=(7,), time=self.obs_seq_len)

        # ee_transform
        transform = traj["obs"]["gripper_cam"]["frames"]["dynamic_extrinsics"]
        assert transform.shape == (T, 4, 4)
        obs_specs["ee_transform"] = ObsSpec(elem_shape=(4, 4), time=self.obs_seq_len)

        # actions
        # concatenate target_joint_pos and target_gripper_pos to get action
        action = ActionSpec(action_dim=8, time=self.action_seq_len)

        self._specs = DataSpecs(
            obs=obs_specs,
            action=action,
        )
