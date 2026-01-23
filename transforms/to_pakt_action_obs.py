from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from environments.specs import (
    ActionSpec,
    DataSpecs,
    NestedTensorSpec,
    ObsSpec,
    PointCloudSpec,
)
from transforms.base_transform import ReversibleTransform
from utils.nested import (
    cat_nested,
    flatten_nested_tensor,
    nested_safe_tensordict_unsqueeze,
    unflatten_nested_tensor,
)

log = logging.getLogger(__name__)


class ToPaktActionObsTransform(ReversibleTransform):

    def __init__(
        self,
        specs: DataSpecs,
        tool_points_key: str = "tool_points",
        target_points_key: str = "target_points",
        gripper_points_key: str = "gripper_points",
    ):
        self.tool_points_key = tool_points_key
        self.target_points_key = target_points_key
        self.gripper_points_key = gripper_points_key
        self.window_len = specs.action_seq_len
        self.num_action_points = 5
        self._specs = specs

        self._load_specs()

    def _load_specs(self) -> None:
        obs_specs = dict(self._specs.obs)  # copy obs specs for local modification

        obs_specs["tool_points"] = NestedTensorSpec(time=False)
        obs_specs["target_points"] = NestedTensorSpec(time=False)
        obs_specs["tool_action_points"] = NestedTensorSpec(time=False)

        # ignore action dims related to static mobile platform
        action = ActionSpec(action_dim=3, time=self.window_len)

        goal_specs = {"text": ObsSpec(elem_shape=(), time=None)}

        self._specs = self._specs.replace(obs=obs_specs, action=action, goal=goal_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        robot_action_points = tensordict["action"]  # (T, N_action, 3)
        tool_pcd = tensordict["obs"][self.tool_points_key]  # (T, N_tool, 3)
        target_pcd = tensordict["obs"][self.target_points_key]  # (T, N_target, 3)
        gripper_pcd = tensordict["obs"][self.gripper_points_key]

        num_tool_points = torch.diff(tool_pcd["points"].offsets())
        num_timesteps = self.window_len + 1
        num_batches = tool_pcd["points"].size(0)

        tool_action_timestep_list = []
        tool_action_point_id_list = []
        tool_action_points = []

        tool_obs_points = []

        for idx, b in enumerate(range(tool_pcd["points"].size(0))):
            n_points = num_tool_points[idx].item()
            # Prepare timestep and point ID tensors
            timestep_vector = (
                torch.arange(
                    num_timesteps,
                    device=tool_pcd["points"].device,
                )
                .unsqueeze(0)
                .expand(n_points, -1)
            )  # (N_tool, num_timesteps)

            point_id_vector = (
                torch.arange(
                    n_points,
                    device=tool_pcd["points"].device,
                )
                .unsqueeze(1)
                .expand(-1, num_timesteps)
            )  # (N_tool, num_timesteps)

            # Extract tool observation points
            tool_obs_points.append(tool_pcd["points"][b, :, 0, :])  # (N_tool, 3)

            # Extract tool action points
            visibility = tool_pcd["visibility"][idx]
            visibility[:, 0] = False  # Don't include obs points in action extraction

            action_points = tool_pcd["points"][
                idx, visibility
            ]  # (N_tool_action, num_timesteps, 3)
            action_timestep = timestep_vector[visibility]  # (N_tool_action,)
            action_point_id = point_id_vector[visibility]  # (N_tool_action,)

            tool_action_points.append(
                action_points
            )  # (N_tool_action, num_timesteps, 3)
            tool_action_timestep_list.append(action_timestep)  # (N_tool_action,)
            tool_action_point_id_list.append(action_point_id)  # (N_tool_action,)

        # TOOL ACTION POINTS
        tool_action_timesteps = torch.nested.as_nested_tensor(
            tool_action_timestep_list, layout=torch.jagged
        )  # (B, N_tool_action)
        tool_action_point_ids = torch.nested.as_nested_tensor(
            tool_action_point_id_list, layout=torch.jagged
        )  # (B, N_tool_action)
        tool_action_points = torch.nested.as_nested_tensor(
            tool_action_points, layout=torch.jagged
        )  # (B, N_tool_action, num_timesteps, 3)

        # TOOL OBS POINTS
        tool_obs_points = torch.nested.as_nested_tensor(
            tool_obs_points, layout=torch.jagged
        )  # (B, N_tool, 3)

        # ROBOT ACTION POINTS
        robot_action_windows = self.sliding_windows_pad_last(
            robot_action_points, window=self.window_len
        ).movedim(
            -1, 2
        )  # (B, N_action, window, 3)
        robot_action_timesteps = (
            torch.arange(self.window_len, device=robot_action_points.device)
            .view(1, 1, -1)
            .expand(num_batches, self.num_action_points, self.window_len)
        ) + 1
        robot_action_point_ids = (
            torch.arange(self.num_action_points, device=robot_action_points.device)
            .view(1, -1, 1)
            .expand(num_batches, self.num_action_points, self.window_len)
        )

        # Merge time and point dims into one
        robot_action_windows = robot_action_windows.reshape(
            num_batches, self.num_action_points * self.window_len, 3
        )  # (B, N_action*window, 3)
        robot_action_timesteps = robot_action_timesteps.reshape(
            num_batches, self.num_action_points * self.window_len
        )  # (B, N_action*window)
        robot_action_point_ids = robot_action_point_ids.reshape(
            num_batches, self.num_action_points * self.window_len
        )  # (B, N_action*window)

        # ROBOT OBS POINTS
        gripper_obs_points = gripper_pcd["points"]  # (B, N_gripper, 3)

        action = cat_nested(
            [robot_action_windows, tool_action_points], dim=1
        )  # (B, N_action + N_tool_points, window, 3)

        ref_actions = self.sliding_windows_pad_last(
            tensordict["ref_action"], window=self.window_len
        ).movedim(
            1, 2
        )  # (B,  window, action_dim)

        obs = {
            "tool_points": {
                "points": tool_obs_points,
                "colors": tool_pcd.get("colors", None),
                "features": tool_pcd.get("features", None),
            },
            "target_points": target_pcd,
            "gripper_points": {
                "points": gripper_obs_points,
            },
            "tool_action_points": {
                "timesteps": tool_action_timesteps,
                "point_ids": tool_action_point_ids,
            },
            "robot_action_points": {
                "timesteps": robot_action_timesteps,
                "point_ids": robot_action_point_ids,
            },
        }

        tensordict["obs"] = tensordict["obs"].update(obs)
        tensordict["action"] = action
        tensordict["phantom_action"] = action.clone()
        tensordict["ref_action"] = ref_actions.squeeze(0)
        return tensordict

    def call_trajectory_rollout(self, tensordict: TensorDict) -> TensorDict:
        tool_pcd = tensordict["obs"][self.tool_points_key]  # (T, N_tool, 3)
        target_pcd = tensordict["obs"][self.target_points_key]  # (T, N_target, 3)
        gripper_points = tensordict["obs"]["gripper_points"]

        tool_pcd = {key: tool_pcd[key].squeeze(0) for key in tool_pcd.keys()}
        target_pcd = {key: target_pcd[key].squeeze(0) for key in target_pcd.keys()}
        gripper_points = {"points": gripper_points["points"].squeeze(0)}

        num_tool_points = tool_pcd["points"].shape[0]
        num_timesteps = self.window_len
        device = tool_pcd["points"].device

        tool_action_timestep = (
            torch.arange(num_timesteps, device=device)
            .unsqueeze(0)
            .expand(num_tool_points, -1)
        ).reshape(
            -1
        )  # (N_tool, window_len)
        tool_action_point_id = (
            torch.arange(num_tool_points, device=device)
            .unsqueeze(1)
            .expand(-1, num_timesteps)
        ).reshape(
            -1
        )  # (N_tool, window_len)

        robot_action_timestep = (
            torch.arange(num_timesteps, device=device)
            .unsqueeze(0)
            .expand(self.num_action_points, -1)
        ).reshape(
            -1
        )  # (N_action, window_len)
        robot_action_point_id = (
            torch.arange(self.num_action_points, device=device)
            .unsqueeze(1)
            .expand(-1, num_timesteps)
        ).reshape(
            -1
        )  # (N_action, window_len)

        obs = {
            "tool_points": tool_pcd,
            "target_points": target_pcd,
            "gripper_points": gripper_points,
            "tool_action_points": {
                "timesteps": tool_action_timestep + 1,
                "point_ids": tool_action_point_id,
            },
            "robot_action_points": {
                "timesteps": robot_action_timestep + 1,
                "point_ids": robot_action_point_id,
            },
        }

        phantom_action = torch.zeros(
            ((self.num_action_points + num_tool_points) * num_timesteps, 3),
            device=device,
        )

        tensordict["obs"] = tensordict["obs"].update(obs)
        tensordict["phantom_action"] = phantom_action
        return tensordict

    def call_trajectory_reverse(self, tensordict: TensorDict) -> TensorDict:
        """
        Inverse of call_trajectory() w.r.t. tensordict["action"]:

        Extracts the original per-timestep robot_action_points (T, N_action, 3)
        from the packed/flattened tensordict["action"] and stores it back into
        tensordict["action"].
        """
        action = tensordict["action"]
        action = torch.stack(
            [a[: self.num_action_points * self.window_len] for a in action.unbind()]
        )

        tensordict["action"] = action.view(
            -1, self.num_action_points, self.window_len, 3
        )  # (T, N_action, window_len, 3)
        return tensordict

    def reverse(self, tensordict):
        return self.call_trajectory_reverse(tensordict)

    def sliding_windows_pad_last(self, x: torch.Tensor, window: int) -> torch.Tensor:
        """
        x: [T, ...]
        returns: [T, window, ...] windows over dim=0, padding the end
                by repeating the last valid timestep.
        """
        assert x.dim() >= 1
        T = x.size(0)
        assert window >= 1

        # pad on the end along time so we can still start a window at t=T-1
        pad_len = window - 1
        if pad_len > 0:
            # F.pad pads last dimension(s), so we temporarily move time to the last dim
            x_last = x.movedim(0, -1)  # [..., T]
            x_last = F.pad(x_last, (0, pad_len), mode="replicate")  # [..., T+pad_len]
            x = x_last.movedim(-1, 0)  # [T+pad_len, ...]

        # make all windows
        # [T+pad_len, ...] -> [T, window, ...]
        x_windows = x.unfold(dimension=0, size=window, step=1)  # [T, window, ...]
        return x_windows

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        return nested_safe_tensordict_unsqueeze(
            self.call_trajectory_rollout(tensordict[0]), dim=0
        )
