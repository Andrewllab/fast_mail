from __future__ import annotations

from typing import Sequence

import open3d as o3d
import open3d.core as o3c
import torch
from tensordict import NonTensorData, TensorDict
from torch_geometric.data import Data

from environments.specs import DataSpecs, ObsSpec
from transforms.base_transform import ReversibleTransform, Transform


class LocalizePAKT(Transform):
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
        obs_specs["localization_mean"] = ObsSpec(3)
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def call_trajectory(self, traj: TensorDict) -> Data:
        mean_point = traj["obs"][self.localization_target].pos.mean(dim=0, keepdim=True)

        traj["obs"]["gripper_points"].pos -= mean_point
        traj["obs"]["tool_points"].pos -= mean_point
        traj["obs"]["target_points"].pos -= mean_point
        traj["action"] -= mean_point
        traj["ref_action"] -= mean_point

        traj["obs"]["localization_mean"] = mean_point

        return traj

    def reverse(self, tensordict: TensorDict) -> TensorDict:
        tensordict["action"] += tensordict["obs"]["localization_mean"]
        tensordict["ref_action"] += tensordict["obs"]["localization_mean"]

        return tensordict
