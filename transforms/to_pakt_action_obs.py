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
from transforms.base_transform import Transform, TransformConstraint
from utils.nested import cat_nested, flatten_nested_tensor

log = logging.getLogger(__name__)


class ToPaktActionObsTransform(Transform):

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
        obs_specs = {}

        obs_specs["tool_points"] = NestedTensorSpec(time=False)
        obs_specs["target_points"] = NestedTensorSpec(time=False)
        obs_specs["gripper_points"] = NestedTensorSpec(time=False)

        # ignore action dims related to static mobile platform
        action = ActionSpec(action_dim=3, time=self.window_len)

        goal_specs = {"text": ObsSpec(elem_shape=(), time=None)}

        self._specs = DataSpecs(obs=obs_specs, action=action, goal=goal_specs)

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
        )

        tool_action_points = [out[:, 1:] for out in tool_pcd["points"].unbind()]
        tool_action_points = torch.nested.as_nested_tensor(
            tool_action_points, layout=torch.jagged
        )

        robot_action_windows = self.sliding_windows_pad_last(
            robot_action_points, window=self.window_len
        )  # (T, window, N_action, 3)

        action = cat_nested(
            [robot_action_windows, tool_action_points], dim=1
        )  # (T, N_action + N_tool, 3)
        action, _, _ = flatten_nested_tensor(
            action, start_dim=1, end_dim=2
        )  # (T, (N_action + N_tool)*N_timesteps, 3)

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
        tensordict["ref_action"] = action.clone()
        return tensordict

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
        x_windows = x_windows.movedim(-1, 2)  # [T, window, ...]
        return x_windows
