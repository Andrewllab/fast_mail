from __future__ import annotations

from typing import Sequence

import torch
from tensordict import NonTensorData, TensorDict
from torch_geometric.data import Data

from environments.specs import DataSpecs, ObsSpec
from transforms.base_transform import ReversibleTransform, Transform


class LocalizePAKT(ReversibleTransform):
    def __init__(
        self,
        specs: DataSpecs,
        localization_target: str,
        backup_localization_target: str | None = None,
        obs_keys: list[str] = [
            "current_points",
            "gripper_points",
            "tool_points",
            "target_points",
            "des_gripper_points",
        ],
        action_keys: list[str] = ["action"],
    ) -> None:

        self.localization_target = localization_target
        self.backup_localization_target = backup_localization_target
        if localization_target not in specs.obs:
            raise ValueError(
                f"Localization target {localization_target} not found in specs.obs"
            )
        if (
            backup_localization_target is not None
            and backup_localization_target not in specs.obs
        ):
            raise ValueError(
                f"Backup localization target {backup_localization_target} not found in specs.obs"
            )

        self.obs_keys = obs_keys
        if isinstance(self.obs_keys, str):
            self.obs_keys = [self.obs_keys]
        self.action_keys = action_keys
        if isinstance(self.action_keys, str):
            self.action_keys = [self.action_keys]

        obs_specs = dict(specs.obs)
        obs_specs["localization_mean"] = ObsSpec((1, 3))
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def calculate_mean_point(self, traj: TensorDict) -> torch.Tensor:
        local_points = traj["obs", self.localization_target, "points"]
        if local_points.ndim == 4:
            mean_point = local_points.mean(axis=1)[:, 0]
        else:
            mean_point = traj["obs"][self.localization_target]["points"].mean(
                dim=1, keepdim=True
            )

        if (
            torch.isnan(mean_point).any()
            and self.backup_localization_target is not None
        ):
            backup_local_points = traj["obs", self.backup_localization_target, "points"]

            if backup_local_points.ndim == 4:
                backup_mean_point = backup_local_points.mean(dim=1)[:, 0:1, :]
            else:
                backup_mean_point = traj[
                    "obs", self.backup_localization_target, "points"
                ].mean(dim=1, keepdim=True)

            mask = torch.isnan(mean_point)
            mean_point = torch.where(mask, backup_mean_point, mean_point)

        return mean_point

    def call_trajectory(self, traj: TensorDict) -> Data:
        B = traj.batch_size[0]

        mean_point = self.calculate_mean_point(traj)

        for key in self.obs_keys:
            if key not in traj["obs"]:
                continue
            if traj["obs", key, "points"].ndim == 4:
                traj["obs", key, "points"] -= mean_point.view(B, 1, 1, 3)
            elif traj["obs", key, "points"].ndim == 3:
                traj["obs", key, "points"] -= mean_point.view(B, 1, 3)

        for key in self.action_keys:
            if key not in traj:
                continue
            if traj[key].ndim == 3 and traj[key].shape[2] == 3:
                traj[key] -= mean_point.view(B, 1, 3)
            elif traj[key].ndim == 4 and traj[key].shape[3] == 3:
                traj[key] -= mean_point.view(B, 1, 1, 3)

        traj["obs"]["localization_mean"] = mean_point

        return traj

    def __call__(self, tensordict):
        return self.call_trajectory(tensordict)

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        mean_point = tensordict["obs"]["localization_mean"]
        B = tensordict.batch_size[0]

        if not hasattr(self, "obs_keys"):
            self.obs_keys = [
                "current_points",
                "gripper_points",
                "tool_points",
                "target_points",
                "des_gripper_points",
            ]
        if not hasattr(self, "action_keys"):
            self.action_keys = ["action"]

        for key in self.obs_keys:
            if key not in tensordict["obs"]:
                continue
            if tensordict["obs", key, "points"].ndim == 4:
                tensordict["obs", key, "points"] += mean_point.view(B, 1, 1, 3)
            elif tensordict["obs", key, "points"].ndim == 3:
                tensordict["obs", key, "points"] += mean_point.view(B, 1, 3)

        for key in self.action_keys:
            if key not in tensordict:
                continue
            if tensordict[key].ndim == 3 and tensordict[key].shape[2] == 3:
                tensordict[key] += mean_point.view(B, 1, 3)
            elif tensordict[key].ndim == 4 and tensordict[key].shape[3] == 3:
                tensordict[key] += mean_point.view(B, 1, 1, 3)

        return tensordict
