import logging
from pathlib import Path
import json

import h5py
import torch
from tensordict import TensorDict

from environments.base_dataset import TrajectoryDataset, keyfunc
from environments.specs import (
    ActionSpec,
    CameraSpec,
    DataSpecs,
    DepthStream,
    PointMapStream,
    ObsSpec,
    PinholeCameraIntrinsic,
    RGBStream,
)
from utils.math import make_pose, matrix_from_quat

log = logging.getLogger(__name__)


class RoboCasaDataset(TrajectoryDataset):
    def __init__(
        self,
        *args,
        trajs_per_task: int | float | None = None,
        **kwargs,
    ):
        self._specs = None
        self.trajs_per_task = trajs_per_task

        super().__init__(*args, **kwargs)

    def find_raw_files(self) -> list[Path]:
        files =  list(self.root_dir.glob("**/*.hdf5"))

        if not files:
            raise FileNotFoundError(
                f"No raw files found in {self.root_dir}. Please check the path."
            )

        files = list(sorted(files, key=keyfunc))
        return files

    def load_from_raw_file(self, filepath: Path) -> TensorDict | list[TensorDict]:
        log.debug(f"Loading trajectories from file {filepath}")

        all_trajs = TensorDict.from_h5(str(filepath))
        # each trajectory is stored under a key like "demo_1", "demo_2", etc.
        demo_keys = list(
            sorted(all_trajs["data"].keys(), key=lambda demo_i: int(demo_i[5:]))
        )
        # required to extract task desriptions from each demonstration as RoboCasa's task descriptions are not unique: https://robocasa.ai/docs/tasks_scenes_assets/atomic_tasks.html
        all_traj_data = h5py.File(str(filepath), "r")

        if isinstance(self.trajs_per_task, float):
            # if trajs_per_task is a fraction, take that fraction of the total
            # number of trajectories
            end = int(len(demo_keys) * self.trajs_per_task)
        else:
            end = self.trajs_per_task

        trajs = []
        for key in demo_keys[:end]:
            traj = all_trajs["data"][key]
            curr_traj = all_traj_data["data"][key]

            robot_state = torch.cat(
                (
                    traj["obs", "robot0_joint_pos"], # shape: (T, 7), robot joint positions
                    traj["obs", "robot0_gripper_qpos"] # shape (T, 2), gripper joint positions which corresponds to the degree of open/close of the gripper (same as gripper_closure in Isaac)
                ), 
                dim=-1
            )

            ee_pose = torch.cat(
                (
                    traj["obs", "robot0_eef_pos"],  # shape: (T, 3), float64
                    traj["obs", "robot0_eef_quat"],  # shape: (T, 4), float64
                ),
                dim=-1,
            )

            traj = TensorDict(
                {
                    "obs": {
                        "left_cam": {
                            "rgb": traj["obs", "robot0_agentview_left_image"], # shape: (T, H, W, 3), uint8
                            "depth": traj["obs", "robot0_agentview_left_depth"].squeeze(-1), # shape: (T, H, W), float32
                            "pointmap": traj["obs", "point_cloud"][:, 0, :, :, :],  # shape (T, H, W, 3), float32
                        },
                        
                        "right_cam": {
                            "rgb": traj["obs", "robot0_agentview_right_image"],  # shape: (T, H, W, 3), uint8
                            "depth": traj["obs", "robot0_agentview_right_depth"].squeeze(-1), # shape: (T, H, W), float32
                            "pointmap": traj["obs", "point_cloud"][:, 1, :, :, :],  # shape (T, H, W, 3), float32
                        },

                        "gripper_cam": {
                            "rgb": traj["obs", "robot0_eye_in_hand_image"],  # shape: (T, H, W, 3), uint8
                            "depth": traj["obs", "robot0_eye_in_hand_depth"].squeeze(-1), # shape: (T, H, W), float32
                            "pointmap": traj["obs", "point_cloud"][:, 2, :, :, :], # shape (T, H, W, 3), float32
                        },
                        "ee_pose": ee_pose, # shape: (T, 7), float64
                        "robot_state": robot_state.float(), # shape: (T, 9), float64
                    },
                    "action": traj["actions"].float()[:, :7], # remove joint dims related to static mobile platform
                    "goal": {
                        "text": json.loads(curr_traj.attrs["ep_meta"])["lang"]  # language description of the current task
                    }
                },  # type: ignore
            )

            trajs.append(traj)

        return trajs

    def _load_specs(self, data: TensorDict | None = None) -> None:
        if data is None:
            filepath = self.find_raw_files()[0]
            log.debug(
                f"Inferring dataset specs by inspecting trajectory from file {filepath}"
            )
            data = TensorDict.from_h5(str(filepath))
            data = data["data", "demo_1"]

        # static left camera
        rgb_shape = data["obs", "robot0_agentview_left_image"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[-1] == 3
        depth_shape = data["obs", "robot0_agentview_left_depth"].shape
        assert len(depth_shape) == 4
        assert depth_shape[-1] == 1
        assert rgb_shape[:-1] == depth_shape[:-1]
        height, width, channels = rgb_shape[1:]

        # TODO: check if extrinsics are in the world frame or ros? 
        # pre-defined camera parameters used by Atalay for demonstration recording: https://github.com/ALRhub/custom_robocasa/blob/main/custom_robocasa/robocasa/utils/camera_utils.py
        intrinsics = torch.zeros(3, 3)  # TODO: fill in correct intrinsics
        left_cam_pos = torch.tensor([[-0.5, 0.35, 1.05]])
        left_cam_rot_quat = torch.tensor([[0.55623853, 0.29935253, -0.37678665, -0.6775092]])  # w, x, y, z
        left_cam_rot_mat = matrix_from_quat(left_cam_rot_quat)
        extrinsics = make_pose(left_cam_pos, left_cam_rot_mat).squeeze(0)
        # # correct extrinsics by adding conversion from ROS to WORLD camera convention
        # # we right-multiply, since we first need to transform the points
        # # into the WORLD convention, and then apply the extrinsics
        # extrinsics[:3, :3] = extrinsics[:3, :3] @ torch.tensor(
        #     ROS_TO_WORLD, dtype=extrinsics.dtype
        # )
        left_cam = CameraSpec(
            streams={
                "rgb": RGBStream(height, width, channels, channel_order="HWC"),
                "depth": DepthStream(height, width, orthogonal=True),
                "pointmap": PointMapStream(height, width, channels=3, channel_order="HWC"),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=height, width=width
            ),
            extrinsics=extrinsics,
        )

        # static right camera
        rgb_shape = data["obs", "robot0_agentview_right_image"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[-1] == 3
        depth_shape = data["obs", "robot0_agentview_right_depth"].shape
        assert len(depth_shape) == 4
        assert depth_shape[-1] == 1
        assert rgb_shape[:-1] == depth_shape[:-1]
        height, width, channels = rgb_shape[1:]

        # TODO: check if extrinsics are in the world frame or ros? 
        # pre-defined camera parameters used by Atalay for demonstration recording: https://github.com/ALRhub/custom_robocasa/blob/main/custom_robocasa/robocasa/utils/camera_utils.py
        intrinsics = torch.zeros(3, 3)  # TODO: fill in correct intrinsics
        right_cam_pos = torch.tensor([[-0.5, -0.35, 1.05]])
        right_cam_rot_quat = torch.tensor([[0.6775091886520386, 0.3767866790294647, -0.2993525564670563, -0.55623859167099]])  # w, x, y, z
        right_cam_rot_mat = matrix_from_quat(right_cam_rot_quat)
        extrinsics = make_pose(right_cam_pos, right_cam_rot_mat).squeeze(0)
        # # correct extrinsics by adding conversion from ROS to WORLD camera convention
        # # we right-multiply, since we first need to transform the points
        # # into the WORLD convention, and then apply the extrinsics
        # extrinsics[:3, :3] = extrinsics[:3, :3] @ torch.tensor(
        #     ROS_TO_WORLD, dtype=extrinsics.dtype
        # )
        right_cam = CameraSpec(
            streams={
                "rgb": RGBStream(height, width, channels, channel_order="HWC"),
                "depth": DepthStream(height, width, orthogonal=True),
                "pointmap": PointMapStream(height, width, channels=3, channel_order="HWC"),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=height, width=width
            ),
            extrinsics=extrinsics,
        )

        # gripper camera
        rgb_shape = data["obs", "robot0_eye_in_hand_image"].shape
        assert len(rgb_shape) == 4
        assert rgb_shape[-1] == 3
        depth_shape = data["obs", "robot0_eye_in_hand_depth"].shape
        assert len(depth_shape) == 4
        assert depth_shape[-1] == 1
        assert rgb_shape[:-1] == depth_shape[:-1]
        height, width, channels = rgb_shape[1:]

        # TODO: check if extrinsics are in the world frame or ros? 
        intrinsics = torch.zeros(3, 3)  # TODO: fill in correct intrinsics
        # pre-defined camera parameters used by Atalay for demonstration recording: https://github.com/ALRhub/custom_robocasa/blob/main/custom_robocasa/robocasa/utils/camera_utils.py
        pos_gripper_cam = torch.tensor([[0.05, 0, 0]])
        rot_quat_gripper_cam = torch.tensor([[0, 0.707107, 0.707107, 0]])  # w, x, y, z
        rot_mat_gripper_cam = matrix_from_quat(rot_quat_gripper_cam)
        extrinsics = make_pose(pos_gripper_cam, rot_mat_gripper_cam).squeeze(0)
        # # correct extrinsics by adding conversion from ROS to WORLD camera convention
        # # we right-multiply, since we first need to transform the points
        # # into the WORLD convention, and then apply the extrinsics
        # extrinsics[:3, :3] = extrinsics[:3, :3] @ torch.tensor(
        #     ROS_TO_WORLD, dtype=extrinsics.dtype
        # )
        gripper_cam = CameraSpec(
            streams={
                "rgb": RGBStream(height, width, channels, channel_order="HWC"),
                "depth": DepthStream(height, width, orthogonal=True),
                "pointmap": PointMapStream(height, width, channels=3, channel_order="HWC"),
            },
            time=self.obs_seq_len,
            intrinsics=PinholeCameraIntrinsic.from_intrinsic_matrix(
                intrinsics, height=height, width=width
            ),
            dynamic_pose_obs_key="gripper_cam_transform",
            # gripper_cam_transform provides complete transform to camera
            extrinsics=extrinsics,
        )

        # robot state
        joint_pos = data["obs", "robot0_joint_pos"]
        assert joint_pos.ndim == 2
        assert joint_pos.shape[-1] == 7
        gripper_pos = data["obs", "robot0_gripper_qpos"]
        assert gripper_pos.ndim == 2
        assert gripper_pos.shape[-1] == 2
        # we concatenate joint_pos and gripper_pos to get a shape of (T, 9)
        robot_state = ObsSpec(elem_shape=(9,), time=self.obs_seq_len)

        # end-effector pose
        ee_pos = data["obs", "robot0_eef_pos"]
        assert ee_pos.ndim == 2
        assert ee_pos.shape[-1] == 3
        ee_quat = data["obs", "robot0_eef_quat"]
        assert ee_quat.ndim == 2
        assert ee_quat.shape[-1] == 4
        # we concatenate ee_pos and ee_quat to get a shape of (T, 7)
        ee_pose = ObsSpec(elem_shape=(7,), time=self.obs_seq_len)
        target_ee_pose = ObsSpec(elem_shape=(7,), time=self.action_seq_len)

        # gripper_cam_transform
        transform = torch.zeros(1, 16)  # TODO: fill in correct transform
        assert transform.ndim == 2
        assert transform.shape[-1] == 16  # flattened 4x4 matrix
        gripper_cam_transform = ObsSpec(elem_shape=(4, 4), time=self.obs_seq_len)

        # !!! NOTE: JOINT-SPACE control actions !!!
        assert data["actions"].ndim == 2
        # ignore joint dims related to static mobile platform
        assert data["actions"][:, :7].shape[-1] == 7
        action = ActionSpec(action_dim=7, time=self.action_seq_len)

        self._specs = DataSpecs(
            obs={
                "left_cam": left_cam,
                "right_cam": right_cam,
                "gripper_cam": gripper_cam,
                "robot_state": robot_state,
                "ee_pose": ee_pose,
                "target_ee_pose": target_ee_pose,
                "gripper_cam_transform": gripper_cam_transform,
            },
            action=action,
        )

    def get_specs(self) -> DataSpecs:
        if self._specs is None:
            self._load_specs()
        assert self._specs is not None
        return self._specs
