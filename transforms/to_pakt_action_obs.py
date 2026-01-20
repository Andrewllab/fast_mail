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
    ):
        self.tool_points_key = tool_points_key
        self.target_points_key = target_points_key
        self.window_len = specs.action_seq_len
        self._specs = specs

        self._load_specs()

    def _load_specs(self) -> None:
        obs_specs = dict(self._specs.obs)  # copy obs specs for local modification

        obs_specs["tool_points"] = NestedTensorSpec(time=False)
        obs_specs["target_points"] = NestedTensorSpec(time=False)
        obs_specs["gripper_points"] = NestedTensorSpec(time=False)

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

        # filter tool points, all need to be visible all timesteps
        conv_tool_pcd = {key: [] for key in tool_pcd.keys()}

        visibility = tool_pcd["visibility"]
        for b in range(visibility.size(0)):
            mask = visibility[b].all(dim=-1)
            for key in tool_pcd.keys():
                conv_tool_pcd[key].append(tool_pcd[key][b][mask])

        for key in conv_tool_pcd.keys():
            conv_tool_pcd[key] = torch.nested.as_nested_tensor(
                conv_tool_pcd[key], layout=torch.jagged
            )
        tool_pcd = conv_tool_pcd

        # split tool points into obs and action
        tool_obs_points = [out[:, 0] for out in tool_pcd["points"].unbind()]
        tool_obs_points = torch.nested.as_nested_tensor(
            tool_obs_points, layout=torch.jagged
        )  # (B, N_tool_points, 3)

        tool_action_points = [out[:, 1:] for out in tool_pcd["points"].unbind()]
        tool_action_points = torch.nested.as_nested_tensor(
            tool_action_points, layout=torch.jagged
        )  # (B, N_tool_points, window, 3)

        robot_action_windows = self.sliding_windows_pad_last(
            robot_action_points, window=self.window_len
        ).movedim(
            -1, 2
        )  # (B, N_action, window, 3)

        action = cat_nested(
            [robot_action_windows, tool_action_points], dim=1
        )  # (B, N_action + N_tool_points, window, 3)
        action, orig_offsets, orig_vshape = flatten_nested_tensor(
            action, start_dim=1, end_dim=2
        )  # (T, (N_action + N_tool)*N_timesteps, 3)

        ref_actions = self.sliding_windows_pad_last(
            tensordict["ref_action"], window=self.window_len
        ).movedim(
            1, 2
        )  # (B,  window, action_dim)

        obs = {
            "tool_points": {
                "points": tool_obs_points,
                "visibility": tool_pcd["visibility"],
                "colors": tool_pcd.get("colors", None),
                "features": tool_pcd.get("features", None),
            },
            "target_points": target_pcd,
            "gripper_points": {"points": robot_action_points},
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

        obs = {
            "tool_points": tool_pcd,
            "target_points": target_pcd,
            "gripper_points": gripper_points,
        }

        num_tool_points = tool_pcd["points"].shape[0]
        num_timesteps = self.window_len
        action_dim = tensordict["obs"]["gripper_points"]["points"].shape[0]
        phantom_action = torch.zeros(
            ((action_dim + num_tool_points) * num_timesteps, 3),
            device=tool_pcd["points"].device,
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
        action = unflatten_nested_tensor(
            action,
            orig_vshape=(self.window_len,),
            start_dim=1,
            end_dim=2,
        )  # (T, N_action + N_tool_points, window_len, 3)

        # Prefer deriving N_action from existing obs if available (most robust).
        n_action = 5  # TODO: get from specs or config
        robot_flat = torch.stack(
            [a[:n_action] for a in action.unbind()],
            dim=0,
        )  # (B, N_action, 20, 3)

        tensordict["action"] = robot_flat  # (T, N_action, 3)
        return tensordict

    # def call_trajectory_reverse(self, tensordict: TensorDict) -> TensorDict:
    #     action = tensordict["action"]  # (T, K, 3) expected

    #     window_len = self.window_len

    #     # So you MUST have N_action available from specs/config, otherwise reverse is ambiguous.
    #     n_action = 5  # TODO: get from specs or config

    #     # 1) Take just the gripper prefix
    #     n_gripper_flat = window_len * n_action

    #     # Case 1: NestedTensor (common if you used jagged tool points)
    #     if isinstance(action, torch.Tensor) and action.is_nested:
    #         # action is shape like: (T, *, 3) but jagged in dim=1 per timestep
    #         # We extract the first n_gripper_flat rows (the gripper prefix) for each timestep.
    #         robot_flat_list = []
    #         for a in action.unbind():  # each `a` is a dense tensor of shape (K_t, 3)
    #             robot_flat_list.append(
    #                 a[:n_gripper_flat, :]
    #             )  # (window_len*n_action, 3)

    #         # Stack back to dense (this should work if n_gripper_flat is constant, which it is)
    #         robot_flat = torch.stack(
    #             robot_flat_list, dim=0
    #         )  # (T, window_len*n_action, 3)

    #     else:
    #         # Case 2: regular dense tensor
    #         # action: (T, K, 3)
    #         robot_flat = action[:, :n_gripper_flat, :]

    #     # 2) Unflatten into windows
    #     robot_win = robot_flat.reshape(
    #         -1, window_len, n_action, 3
    #     )  # (T, window_len, n_action, 3)

    #     # 3) Invert the sliding-window transform: pick the "current timestep" element
    #     # robot_action_points = robot_win[:, 0, :, :]  # (T, n_action, 3)

    #     tensordict["action"] = robot_win
    #     return tensordict

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
