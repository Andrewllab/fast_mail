from __future__ import annotations

from typing import Sequence

from tensordict import TensorDict

from environments.specs import DataSpecs, PointCloudSpec
from transforms.base_transform import Transform


class CenterPointCloudTrajectory(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        transform_obs_key: str,
        xy_only: bool = True,
        pcd_keys: str | Sequence[str] = "pcd",
    ) -> None:

        if isinstance(pcd_keys, str):
            pcd_keys = [pcd_keys]
        else:
            pcd_keys = list(pcd_keys)

        for key in pcd_keys:
            if key not in specs.obs:
                raise ValueError(f"Point cloud key '{key}' not found in specs.")
            if not isinstance(specs.obs[key], PointCloudSpec):
                raise ValueError(f"Key '{key}' is not a PointCloudSpec.")

        self.transform_obs_key = transform_obs_key
        self.xy_only = xy_only
        self._pcd_keys = pcd_keys
        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:

        transform = tensordict["obs", self.transform_obs_key]

        # use the translation part of the first transform in the trajectory as the center
        center = transform[0, :3, 3]  # (3,)

        if self.xy_only:
            center[2] = 0.0

        for key in self._pcd_keys:
            data = tensordict["obs", key]
            assert data.pos is not None

            data.pos[...] -= center

        return tensordict
