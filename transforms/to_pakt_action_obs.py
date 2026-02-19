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
from transforms.base_transform import ReversibleTransform, Transform
from utils.nested import cat_nested

log = logging.getLogger(__name__)

TOKEN_TARGET_TYPE = 0
TOKEN_TOOL_TYPE = 1
TOKEN_GRIPPER_TYPE = 2
TOKEN_DES_GRIPPER_TYPE = 3


class ToPaktActionObsTransform(ReversibleTransform):

    def __init__(
        self,
        specs: DataSpecs,
        tracked_points_keys: str = "tool_points",
        untracked_points_keys: str = "target_points",
        gripper_points_key: str = "gripper_points",
        des_gripper_points_key: str = "des_gripper_points",
        include_tracked_in_action: bool = True,
    ):
        # Environment point clouds
        self.tracked_points_keys = tracked_points_keys
        if isinstance(self.tracked_points_keys, str):
            self.tracked_points_keys = [self.tracked_points_keys]
        self.untracked_points_keys = untracked_points_keys
        if isinstance(self.untracked_points_keys, str):
            self.untracked_points_keys = [self.untracked_points_keys]

        # Gripper point clouds for obs only, not action
        self.gripper_points_key = gripper_points_key
        self.des_gripper_points_key = des_gripper_points_key

        self.include_tracked_in_action = include_tracked_in_action
        self.window_len = specs.action_seq_len
        self.num_action_points = 5
        self._specs = specs

        self._load_specs()

    def _load_specs(self) -> None:
        obs_specs = dict(self._specs.obs)  # copy obs specs for local modification

        obs_specs["current_points"] = NestedTensorSpec(time=False)
        obs_specs["action_points"] = NestedTensorSpec(time=False)

        # ignore action dims related to static mobile platform
        action = ActionSpec(action_dim=3, time=self.window_len)

        goal_specs = {"text": ObsSpec(elem_shape=(), time=None)}

        self._specs = self._specs.replace(obs=obs_specs, action=action, goal=goal_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        tracked_pcds = [tensordict["obs"][k] for k in self.tracked_points_keys]
        untracked_pcds = [tensordict["obs"][k] for k in self.untracked_points_keys]
        gripper_pcd = tensordict["obs"][self.gripper_points_key]
        des_gripper_pcd = tensordict["obs"][self.des_gripper_points_key]
        action_pcds = tensordict["action"]

        B = action_pcds.size(0)
        device = action_pcds.device

        # Split tracked pcds into current (obs) and future (action)
        obj_id_counter = 2
        tracked_current_pcds = []
        tracked_future_pcds = []
        for pcd in tracked_pcds:
            # Split into current and future, add object ids, and store in lists for later concatenation
            current_pcd, future_pcd = self._split_tracked(pcd, obj_id=obj_id_counter)

            # Add object ids to current and future pcds (if not already added in _split_tracked)
            current_pcd = self._add_ids_to_pcd(current_pcd, obj_id=obj_id_counter)
            future_pcd = self._add_ids_to_pcd(future_pcd, obj_id=obj_id_counter)

            tracked_current_pcds.append(current_pcd)
            tracked_future_pcds.append(future_pcd)
            obj_id_counter += 1

        # Add object ids to untracked pcds
        for pcd in untracked_pcds:
            pcd = self._add_ids_to_pcd(pcd, obj_id=obj_id_counter)
            obj_id_counter += 1

        current_pcds = tracked_current_pcds + untracked_pcds

        # Simulate gripper point dictionaries to have the same keys as all current_pcds
        gripper_pcd = self._simulate_pcd_for_gripper(
            gripper_pcd, reference_pcd=current_pcds[0], obj_id_counter=obj_id_counter
        )
        obj_id_counter += 1
        current_pcds.append(gripper_pcd)

        des_gripper_pcd = self._simulate_pcd_for_gripper(
            des_gripper_pcd,
            reference_pcd=current_pcds[0],
            obj_id_counter=obj_id_counter,
        )
        obj_id_counter += 1
        current_pcds.append(des_gripper_pcd)

        tmp_cur_pcd = {}
        for key in current_pcds[0].keys():
            tmp_cur_pcd[key] = cat_nested(
                [cur_pcd[key] for cur_pcd in current_pcds], dim=1
            )
        current_pcds = tmp_cur_pcd

        action_pcds = {"points": action_pcds}
        # Always have object id 1 for gripper action points
        action_pcds = self._simulate_pcd_for_action(
            action_pcds, reference_pcd=current_pcds, obj_id_counter=1
        )

        future_pcds = [action_pcds] + tracked_future_pcds
        tmp_future_pcd = {}
        for key in future_pcds[0].keys():
            tmp_future_pcd[key] = cat_nested(
                [future_pcd[key] for future_pcd in future_pcds], dim=1
            )
        future_pcds = tmp_future_pcd

        action = future_pcds.pop("points")  # (B, N_action*window_len, 3)
        obs = {
            "current_points": current_pcds,
            "action_points": future_pcds,
        }

        tensordict["obs"] = tensordict["obs"].update(obs)
        tensordict["action"] = action
        tensordict["phantom_action"] = action.clone()
        return tensordict

    def call_trajectory_rollout(self, tensordict: TensorDict) -> TensorDict:
        tool_pcd = tensordict["obs"][self.tool_points_key]  # (T, N_tool, 3)
        target_pcd = tensordict["obs"][self.target_points_key]  # (T, N_target, 3)
        gripper_points = tensordict["obs"][self.gripper_points_key]
        des_gripper_points = tensordict["obs"][self.des_gripper_points_key]

        num_timesteps = self.window_len
        B = tool_pcd["points"].size(0)
        device = tool_pcd["points"].device

        # Tool indices + phantom action without per-batch Python loops
        tool_action_timesteps, tool_action_point_ids, phantom_action = (
            self._rollout_tool_indices_and_phantom(tool_pcd["points"])
        )

        robot_action_timesteps = (
            torch.arange(num_timesteps, device=device)
            .view(1, 1, -1)
            .expand(B, self.num_action_points, -1)
        ).reshape(
            B, -1
        )  # (N_action, window_len)
        robot_action_point_ids = (
            torch.arange(self.num_action_points, device=device)
            .view(1, -1, 1)
            .expand(B, -1, num_timesteps)
        ).reshape(
            B, -1
        )  # (N_action, window_len)

        obs = {
            "tool_points": tool_pcd,
            "target_points": target_pcd,
            "gripper_points": gripper_points,
            "des_gripper_points": des_gripper_points,
            "tool_action_points": {
                "timesteps": tool_action_timesteps + 1,
                "point_ids": tool_action_point_ids,
            },
            "robot_action_points": {
                "timesteps": robot_action_timesteps + 1,
                "point_ids": robot_action_point_ids,
            },
        }

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

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if "action" in tensordict.keys():
            return self.call_trajectory(tensordict)
        else:
            return self.call_trajectory_rollout(tensordict)

    def _simulate_pcd_for_gripper(
        self, gripper_pcd: TensorDict, reference_pcd: TensorDict, obj_id_counter: int
    ) -> TensorDict:
        # Create a pcd dict for gripper points with the same keys as current_pcd, filling in zeros where necessary
        gripper_points = gripper_pcd["points"]  # (B, N_gripper, 3)
        B = gripper_points.shape[0]
        N = gripper_points.shape[1]
        device = gripper_points.device

        # Give IDs to each of the gripper points
        gripper_pcd["gripper_ids"] = (
            torch.arange(1, N + 1, device=device).unsqueeze(0).expand(B, -1)
        )
        # Assign a unique object ID to all gripper points (different from any environment object)
        gripper_pcd["object_ids"] = torch.full_like(
            gripper_points[..., 0], obj_id_counter, dtype=torch.long
        )
        gripper_pcd["timesteps"] = torch.zeros_like(gripper_pcd["object_ids"])

        for key in reference_pcd.keys():
            if key not in gripper_pcd.keys():
                gripper_pcd[key] = torch.zeros(
                    (B, N, reference_pcd[key].shape[-1]),
                    dtype=reference_pcd[key].dtype,
                    device=device,
                )

        return gripper_pcd

    def _simulate_pcd_for_action(
        self, action_pcd: TensorDict, reference_pcd: TensorDict, obj_id_counter: int
    ) -> TensorDict:
        # Create a pcd dict for action points with the same keys as current_pcd, filling in zeros where necessary
        action_points = action_pcd["points"]  # (B, N_action, 3)
        B = action_points.shape[0]
        N = action_points.shape[1]
        T = action_points.shape[2]
        device = action_points.device

        action_pcd["object_ids"] = torch.full_like(
            action_points[..., 0], obj_id_counter, dtype=torch.long
        )
        action_pcd["gripper_ids"] = (
            torch.arange(1, N + 1, device=device).reshape(1, -1, 1).expand(B, N, T)
        )
        action_pcd["timesteps"] = (
            torch.arange(T, device=device).reshape(1, 1, -1).expand(B, N, T) + 1
        )

        for key in reference_pcd.keys():
            if key not in action_pcd.keys():
                action_pcd[key] = torch.zeros(
                    (B, N, T, reference_pcd[key].shape[-1]),
                    dtype=reference_pcd[key].dtype,
                    device=device,
                )

        # Flatten the point and time dims together for all keys to match the expected shape of (B, N_action*window_len, feature_dim)
        for key in action_pcd.keys():
            action_pcd[key] = action_pcd[key].reshape(B, N * T, -1)
            if action_pcd[key].shape[2] == 1:
                action_pcd[key] = action_pcd[key].squeeze(2)

        return action_pcd

    def _add_ids_to_pcd(self, pcd: TensorDict, obj_id: int) -> TensorDict:
        # Add object_ids key to pcd with the same ragged structure as points, filled with obj_id
        device = pcd["points"].device
        points_off = pcd["points"].offsets()  # (B+1,)
        points_val = pcd["points"].values()  # (sum_N, 3)

        pcd["object_ids"] = torch.nested.nested_tensor_from_jagged(
            values=torch.full(
                (points_val.size(0),),
                fill_value=obj_id,
                dtype=torch.long,
                device=device,
            ),
            offsets=points_off,
        )

        pcd["gripper_ids"] = torch.zeros_like(
            pcd["object_ids"]
        )  # default gripper id = 0 for all points

        pcd["timesteps"] = torch.zeros_like(
            pcd["object_ids"]
        )  # default timestep = 0 for all points

        return pcd

    def _split_tracked(
        self, pcd: TensorDict, obj_id: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Jagged points: logical shape (B, N_tool, num_timesteps, 3)
        tool_points_nt = pcd["points"]
        points_off = tool_points_nt.offsets()  # (B+1,)
        points_val = tool_points_nt.values()  # (sum_N, num_timesteps, 3)

        device = points_val.device

        if points_val.ndim != 3:
            raise RuntimeError(
                f"Expected 3D points tensor with shape (sum_N, num_timesteps, 3), got {points_val.shape}"
            )

        current_pcd = {
            key: pcd[key]
            for key in pcd.keys()
            if key != "points" and key != "visibility"
        }
        future_pcd = {}

        # Get informations about batch sizes and point counts from offsets
        B = points_off.numel() - 1
        lengths = torch.diff(points_off)  # (B,)

        # Current points: timestep 0 for every tool point, rewrapped with SAME offsets (no Python loop)
        current_points_vals = points_val[:, 0, :]  # (sum_N, 3)
        current_points = torch.nested.nested_tensor_from_jagged(
            values=current_points_vals, offsets=points_off
        )  # (B, N_tool, 3)
        current_pcd["points"] = current_points
        current_pcd["object_ids"] = torch.nested.nested_tensor_from_jagged(
            values=torch.full(
                (points_val.size(0),),
                fill_value=obj_id,
                dtype=torch.long,
                device=device,
            ),
            offsets=points_off,
        )
        current_pcd["timesteps"] = torch.zeros_like(current_pcd["object_ids"])

        # Visibility: logical shape (B, N_tool, num_timesteps)
        vis_nt = pcd["visibility"]
        vis_off = vis_nt.offsets()
        vis_val = vis_nt.values()  # (sum_N, num_timesteps)

        # Sanity: ragged structure must match points' ragged structure
        if vis_off.numel() != points_off.numel() or not torch.equal(
            vis_off, points_off
        ):
            raise RuntimeError(
                "tool_pcd['visibility'] ragged structure must match tool_pcd['points']."
            )

        # Exclude obs timestep (0) from action extraction ONCE (avoid per-batch clone)
        # If visibility is reused elsewhere and must remain unchanged, clone once here.
        vis_work = vis_val.clone()
        vis_work[:, 0] = False

        # Find all selected (flat_point, t) pairs at once
        sel = vis_work.nonzero(
            as_tuple=False
        )  # (K, 2) columns: [flat_point_idx, timestep]

        # If no points selected, return empty jagged tensors
        if sel.numel() == 0 or not self.include_tracked_in_action:
            # Empty offsets for future points
            empty_off = torch.zeros(
                (B + 1,), device=points_off.device, dtype=points_off.dtype
            )

            # Go through all keys
            for key in pcd.keys():
                empty_vals = pcd[key].new_empty(pcd[key].shape[-1])
                if key == "points":
                    empty_vals = empty_vals[None]

                future_pcd[key] = torch.nested.nested_tensor_from_jagged(
                    values=empty_vals, offsets=empty_off
                )

            return (current_pcd, future_pcd)

        flat_p = sel[:, 0]  # (K,)
        t = sel[:, 1]  # (K,)

        # Gather action point coordinates
        tool_action_vals = points_val[flat_p, t, :]  # (K, 3)

        # Map flat point index -> batch id (row id) so we can rewrap as jagged over B
        # point_batch_ids has length sum_N; each tool point knows which batch it belongs to
        point_batch_ids = torch.repeat_interleave(
            torch.arange(B, device=points_off.device), lengths
        )  # (sum_N,)
        b_ids = point_batch_ids[flat_p]  # (K,)

        # Build action jagged offsets from counts per batch
        counts = torch.bincount(b_ids, minlength=B)  # (B,)
        action_off = torch.empty(
            (B + 1,), device=points_off.device, dtype=points_off.dtype
        )
        action_off[0] = 0
        action_off[1:] = counts.cumsum(0)

        # Rewrap outputs as jagged nested tensors over batch
        future_points = torch.nested.nested_tensor_from_jagged(
            values=tool_action_vals, offsets=action_off
        )  # (B, N_tool_action, 3)
        future_pcd["points"] = future_points

        future_timesteps = torch.nested.nested_tensor_from_jagged(
            values=t.to(torch.long), offsets=action_off
        )  # (B, N_tool_action)
        future_pcd["timesteps"] = future_timesteps

        future_object_ids = torch.nested.nested_tensor_from_jagged(
            values=torch.full_like(t, fill_value=obj_id),
            offsets=action_off,
        )  # (B, N_tool_action)
        future_pcd["object_ids"] = future_object_ids

        for key in pcd.keys():
            if key in ["points", "visibility"]:
                continue

            vals = pcd[key].values()
            tool_action_vals = vals[flat_p]  # (K, ...)

            future_pcd[key] = torch.nested.nested_tensor_from_jagged(
                values=tool_action_vals, offsets=action_off
            )

        return (current_pcd, future_pcd)

    def _rollout_tool_indices_and_phantom(
        self,
        tool_points_nt: torch.Tensor,  # jagged NT: (B, N_tool, 3) or (B, N_tool, ...)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
        tool_action_timesteps: jagged NT (B, N_tool*num_timesteps)  (0..T-1) per point
        tool_action_point_ids: jagged NT (B, N_tool*num_timesteps)  (0..N_tool-1) repeated across timesteps
        phantom_action:        jagged NT (B, (num_action_points + N_tool)*num_timesteps, 3) zeros
        """
        assert tool_points_nt.is_nested, "expected jagged NestedTensor for tool points"

        device = tool_points_nt.values().device
        off = tool_points_nt.offsets()  # (B+1,)
        B = off.numel() - 1
        lengths = (off[1:] - off[:-1]).to(torch.long)  # (B,) == N_tool per batch
        sum_N = int(lengths.sum().item())

        # -------- tool_action_* (size per batch = N_tool * T) --------
        # For each tool point (flattened), repeat timesteps 0..T-1
        t = torch.arange(self.window_len, device=device, dtype=torch.long)  # (T,)
        tool_action_timesteps_vals = t.repeat(sum_N)  # (sum_N*T,)

        # For each tool point (flattened), its "local id" within its batch, repeated T times
        # Build local ids in flattened order: [0..N0-1, 0..N1-1, ...]
        # Create global ids 0..sum_N-1 and subtract batch starts.
        global_pid = torch.arange(sum_N, device=device, dtype=torch.long)  # (sum_N,)
        batch_ids = torch.repeat_interleave(
            torch.arange(B, device=device), lengths
        )  # (sum_N,)
        local_pid = global_pid - off[batch_ids].to(torch.long)  # (sum_N,)
        tool_action_point_ids_vals = local_pid.repeat_interleave(
            self.window_len
        )  # (sum_N*T,)

        # -------- phantom_action (size per batch = (num_action_points + N_tool)*T) --------
        if self.include_tool_in_action:
            # Offsets for (B, N_tool*T): multiply by T
            tool_action_off = off.to(torch.long) * self.window_len  # (B+1,)
            tool_action_timesteps = torch.nested.nested_tensor_from_jagged(
                values=tool_action_timesteps_vals, offsets=tool_action_off
            )
            tool_action_point_ids = torch.nested.nested_tensor_from_jagged(
                values=tool_action_point_ids_vals, offsets=tool_action_off
            )

            phantom_lengths = (
                lengths + self.num_action_points
            ) * self.window_len  # (B,)
        else:
            phantom_lengths = (
                self.num_action_points * self.window_len * torch.ones_like(lengths)
            )

            empty_off = torch.zeros((B + 1,), device=device, dtype=torch.long)
            empty_vals1 = torch.zeros((0,), device=device, dtype=torch.long)

            tool_action_timesteps = torch.nested.nested_tensor_from_jagged(
                values=empty_vals1, offsets=empty_off
            )
            tool_action_point_ids = torch.nested.nested_tensor_from_jagged(
                values=empty_vals1, offsets=empty_off
            )

        phantom_off = torch.empty((B + 1,), device=device, dtype=torch.long)
        phantom_off[0] = 0
        phantom_off[1:] = phantom_lengths.cumsum(0)

        phantom_vals = torch.zeros(
            (int(phantom_off[-1].item()), 3),
            device=device,
            dtype=tool_points_nt.values().dtype,
        )
        phantom_action = torch.nested.nested_tensor_from_jagged(
            values=phantom_vals, offsets=phantom_off
        )

        return tool_action_timesteps, tool_action_point_ids, phantom_action


class ToActionSlidingWindowTransform(Transform):
    def __init__(self, specs: DataSpecs):
        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict["action"]  # (T, N_action, action_dim)
        new_action = sliding_windows_pad_last(
            action, window=self._specs.action_seq_len
        )  # (T, N_action, window, action_dim)
        new_action = new_action.permute(0, 2, 1, 3)  # (T, window, N_action, action_dim)

        tensordict["action"] = new_action

        ref_action = tensordict["ref_action"]
        new_ref_action = sliding_windows_pad_last(
            ref_action, window=self._specs.action_seq_len
        )  # (T, window, action_dim)

        tensordict["ref_action"] = new_ref_action

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        return tensordict


def sliding_windows_pad_last(
    x: torch.Tensor, window: int, time_dim: int = 0
) -> torch.Tensor:
    """
    Windows over time_dim, pads by repeating the last timestep.

    Returns a tensor where the new window dimension is inserted right after time_dim.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if x.dim() <= time_dim:
        raise ValueError("time_dim out of range")

    T = x.size(time_dim)
    pad_len = window - 1

    if pad_len > 0:
        # take last timestep slice along time_dim
        index = [slice(None)] * x.dim()
        index[time_dim] = slice(T - 1, T)
        last = x[tuple(index)]  # shape has time_dim size 1

        # expand last along time_dim to pad_len
        expand_shape = list(last.shape)
        expand_shape[time_dim] = pad_len
        last = last.expand(*expand_shape)

        x = torch.cat([x, last], dim=time_dim)

    out = x.unfold(dimension=time_dim, size=window, step=1)
    out = out.movedim(-1, time_dim + 1)
    return out
