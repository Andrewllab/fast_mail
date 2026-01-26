import json
import logging
import os
import re
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
    ObsSpec,
    PinholeCameraIntrinsic,
    RGBStream,
)
from transforms.base_transform import TransformPartialsDict, init_transforms
from utils.math import convert_quat
from utils.paths import iglob_follow_symlinks, resolve_path

log = logging.getLogger(__name__)


class RoboCasaDataset(CustomHdf5Dataset):
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

        # no need to sort
        self.files = list(iglob_follow_symlinks(self._root_dir, "**/*.hdf5"))

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

            file_trajs = h5_file["data"]
            assert isinstance(file_trajs, Group)
            traj_keys = list(sorted(file_trajs.keys(), key=self.traj_name_keyfunc))
            traj_keys = get_subset(traj_keys, load_subset)

            self.trajs.extend(
                [
                    (file.relative_to(self._root_dir), traj_key, file_trajs[traj_key])
                    for traj_key in traj_keys
                ]
            )

        if not self.trajs:
            raise ValueError(f"Subset {load_subset} resulted in zero trajectories.")

        traj_lengths = [len(traj["actions"]) for _, _, traj in self.trajs]

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
        # each trajectory is stored under a key like "demo_1", "demo_2", etc.
        match = re.fullmatch(r"demo_(\d+)", key)
        assert match is not None
        return int(match.group(1))

    def get_trajectory(self, traj_idx: int) -> TensorDict:
        path, name, traj = self.trajs[traj_idx]

        # shape: (T, 7), float32
        joint_pos = traj["obs"]["robot0_joint_pos"][...].astype(np.float32)
        # gripper joint positions which corresponds to the degree of open/close of the gripper (same as gripper_closure in Isaac)
        # shape: (T, 2), float32
        gripper_pos = traj["obs"]["robot0_gripper_qpos"][...].astype(np.float32)
        # shape: (T, 3), float32
        ee_pos = traj["obs"]["robot0_eef_pos"][...].astype(np.float32)
        # shape: (T, 4), float32
        ee_quat = traj["obs"]["robot0_eef_quat_site"][...].astype(np.float32)
        ee_quat = convert_quat(ee_quat, "wxyz")

        base_to_ee_pos = traj["obs"]["robot0_base_to_eef_pos"][...].astype(np.float32)
        base_to_ee_quat = traj["obs"]["robot0_base_to_eef_quat_site"][...].astype(
            np.float32
        )
        base_to_ee_quat = convert_quat(base_to_ee_quat, "wxyz")

        base_to_ee_pose = torch.cat(
            (
                torch.from_numpy(base_to_ee_pos),
                torch.from_numpy(base_to_ee_quat),
            ),
            dim=-1,
        )

        base_pos = traj["obs"]["robot0_base_pos"][...].astype(np.float32)
        base_quat = traj["obs"]["robot0_base_quat"][...].astype(np.float32)
        base_quat = convert_quat(base_quat, "wxyz")
        base_pose = torch.cat(
            (
                torch.from_numpy(base_pos),
                torch.from_numpy(base_quat),
            ),
            dim=-1,
        )

        # remove action dims related to static mobile platform
        action = traj["actions"][:, :7].astype(np.float32)

        action_dict = traj["action_dict"]
        obs_next_actions = {
            f"next_{k}": action_dict[k][...] for k in action_dict.keys()
        }

        # # ensure we are not ignoring any relevant actions
        # assert np.allclose(traj["actions"][:, 7:], np.array([0, 0, 0, 0, -1]))

        robot_state = torch.cat(
            (
                torch.from_numpy(joint_pos),
                torch.from_numpy(gripper_pos),
            ),
            dim=-1,
        )

        ee_pose = torch.cat(
            (
                torch.from_numpy(ee_pos),
                torch.from_numpy(ee_quat),
            ),
            dim=-1,
        )

        camera_poses = traj["camera_params"]["dynamic"]

        ep_meta = json.loads(traj.attrs["ep_meta"])

        # TODO: does it use less memory if we explicitly convert to torch tensors first?
        td_dict = {
            "obs": {
                # cameras
                "left_cam": {
                    # shape: (T, H, W, 3), uint8
                    "rgb": traj["obs"]["robot0_agentview_left_image"][...],
                    # shape: (T, H, W), float32
                    "depth": traj["obs"]["robot0_agentview_left_depth"][..., 0],
                },
                "right_cam": {
                    # shape: (T, H, W, 3), uint8
                    "rgb": traj["obs"]["robot0_agentview_right_image"][...],
                    # shape: (T, H, W), float32
                    "depth": traj["obs"]["robot0_agentview_right_depth"][..., 0],
                },
                "gripper_cam": {
                    # shape: (T, H, W, 3), uint8
                    "rgb": traj["obs"]["robot0_eye_in_hand_image"][...],
                    # shape: (T, H, W), float32
                    "depth": traj["obs"]["robot0_eye_in_hand_depth"][..., 0],
                },
                # camera poses (shape: (T, 4, 4))
                "left_cam_pose": camera_poses["robot0_agentview_left"]["extrinsics"][
                    ...
                ],
                "right_cam_pose": camera_poses["robot0_agentview_right"]["extrinsics"][
                    ...
                ],
                "gripper_cam_pose": camera_poses["robot0_eye_in_hand"]["extrinsics"][
                    ...
                ],
                "joint_pos": joint_pos,  # shape: (T, 7), float32
                "robot_state": robot_state,  # shape: (T, 9), float32
                "ee_pose": ee_pose,  # shape: (T, 7), float32
                "base_to_ee_pose": base_to_ee_pose,  # shape: (T, 7), float32
                "base_pose": base_pose,  # shape: (T, 7), float32
                **obs_next_actions,  # next actions for all action components
            },
            "action": action,
            "ref_action": action.copy(),
            "goal": {
                # language description of the current task
                "text": ep_meta["lang"],
            },
            "path": str(path),
            "name": name,
        }  # type: ignore

        # We assume that if we have the segmentation ids we also have the segmentation masks
        if "segmentation_ids" in traj:
            td_dict["obs"]["left_cam"]["segmentation"] = traj["obs"][
                "robot0_agentview_left_segmentation_element"
            ][...].squeeze(-1)
            td_dict["obs"]["right_cam"]["segmentation"] = traj["obs"][
                "robot0_agentview_right_segmentation_element"
            ][...].squeeze(-1)
            td_dict["obs"]["gripper_cam"]["segmentation"] = traj["obs"][
                "robot0_eye_in_hand_segmentation_element"
            ][...].squeeze(-1)

            td_dict["segmentation_ids"] = {
                key: traj["segmentation_ids"][key][...].astype(np.int32)
                for key in traj["segmentation_ids"].keys()
            }

        td = TensorDict(td_dict)
        # add a batch dimension so we can index
        td["obs"].auto_batch_size_(batch_dims=1)

        return td

    def _load_specs(self) -> None:
        _, _, traj = self.trajs[0]

        obs_specs = {}

        cam_names = ["left_cam", "right_cam", "gripper_cam"]
        raw_keys = [
            "robot0_agentview_left",
            "robot0_agentview_right",
            "robot0_eye_in_hand",
        ]

        for key, cam_name in zip(raw_keys, cam_names):
            # !!! IMPORTANT: "static" cameras are attached to the robot platform which sometimes moves caused by the robot-arm movements, so they move as well !!!

            rgb = traj["obs"][f"{key}_image"]
            match rgb.shape:
                case (T, height, width, 3):
                    pass
                case _:
                    raise ValueError(
                        f"Expected left camera RGB images to have shape (T, H, W, 3), got {rgb.shape}"
                    )

            depth = traj["obs"][f"{key}_depth"]
            match depth.shape:
                case (t, h, w, 1) if t == T and h == height and w == width:
                    pass
                case _:
                    raise ValueError(
                        f"Expected left camera depth images to have shape ({T}, {height}, {width}, 1), got {depth.shape}"
                    )

            intrinsics = traj["camera_params"]["dynamic"][key]["intrinsics"]
            # verify that intrinsics never change over time
            assert intrinsics.shape == (1, 3, 3)

            cam_spec = CameraSpec(
                streams={
                    "rgb": RGBStream(height, width, 3, channel_order="HWC"),
                    "depth": DepthStream(height, width),
                },
                time=self.obs_seq_len,
                intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                    intrinsics[0], height=height, width=width
                ),
                dynamic_pose_obs_key=f"{cam_name}_pose",
                extrinsics=torch.eye(4, dtype=torch.float32),
            )
            obs_specs[cam_name] = cam_spec

            cam_pose = traj["camera_params"]["dynamic"][key]["extrinsics"]
            match cam_pose.shape:
                case (t, 4, 4) if t == T:
                    pass
                case _:
                    raise ValueError(
                        f"Expected {cam_name}_pose to have shape ({T}, 4, 4), got {cam_pose.shape}"
                    )
            obs_specs[f"{cam_name}_pose"] = ObsSpec(
                elem_shape=(4, 4), time=self.obs_seq_len
            )

        # robot state
        joint_pos = traj["obs"]["robot0_joint_pos"]
        assert joint_pos.shape == (T, 7)
        gripper_pos = traj["obs"]["robot0_gripper_qpos"]
        assert gripper_pos.shape == (T, 2)
        # we concatenate joint_pos and gripper_pos to get a shape of (T, 9)
        obs_specs["joint_pos"] = ObsSpec(elem_shape=(7,), time=self.obs_seq_len)
        obs_specs["robot_state"] = ObsSpec(elem_shape=(9,), time=self.obs_seq_len)

        # end-effector pose
        ee_pos = traj["obs"]["robot0_eef_pos"]
        assert ee_pos.shape == (T, 3)
        ee_quat = traj["obs"]["robot0_eef_quat_site"]
        assert ee_quat.shape == (T, 4)
        # we concatenate ee_pos and ee_quat to get a shape of (T, 7)
        obs_specs["ee_pose"] = ObsSpec(elem_shape=(7,), time=self.obs_seq_len)

        assert traj["actions"].shape == (T, 12)
        # ignore action dims related to static mobile platform
        action = ActionSpec(action_dim=7, time=self.action_seq_len)

        goal_specs = {"text": ObsSpec(elem_shape=(), time=None)}

        self._specs = DataSpecs(obs=obs_specs, action=action, goal=goal_specs)
