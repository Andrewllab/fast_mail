from __future__ import annotations

from typing import Sequence

import open3d as o3d
import open3d.core as o3c
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
    ) -> None:

        assert localization_target in [
            "gripper_points",
            "tool_points",
            "target_points",
        ], f"Invalid localization target: {localization_target}"
        self.localization_target = localization_target

        obs_specs = dict(specs.obs)
        obs_specs["localization_mean"] = ObsSpec((1, 3))
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def call_trajectory(self, traj: TensorDict) -> Data:
        mean_point = traj["obs"][self.localization_target]["points"].mean(
            dim=1, keepdim=True
        )

        traj["obs"]["gripper_points"]["points"] -= mean_point
        traj["obs"]["tool_points"]["points"] -= mean_point
        traj["obs"]["target_points"]["points"] -= mean_point
        if "action" in traj:
            traj["action"] -= mean_point

        traj["obs"]["localization_mean"] = mean_point

        return traj

    def __call__(self, tensordict):
        return self.call_trajectory(tensordict)

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        mean_point = tensordict["obs"]["localization_mean"]

        tensordict["obs"]["gripper_points"]["points"] += mean_point
        tensordict["obs"]["tool_points"]["points"] += mean_point
        tensordict["obs"]["target_points"]["points"] += mean_point
        if "action" in tensordict:
            tensordict["action"] += mean_point

        return tensordict
