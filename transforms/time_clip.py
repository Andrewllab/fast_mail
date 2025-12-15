from __future__ import annotations

import logging

from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import Transform, TransformConstraint

log = logging.getLogger(__name__)


class ClipTrajectoryLength(Transform):
    constraints = [TransformConstraint.TRAJECTORY_ONLY]

    def __init__(
        self,
        specs: DataSpecs,
        length: int,
    ):
        self._specs = specs
        self.length = length

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        # TODO: slice anything with a leading dimension of T
        tensordict["obs"] = tensordict["obs"][: self.length]
        tensordict["action"] = tensordict["action"][: self.length]
        tensordict["ref_action"] = tensordict["ref_action"][: self.length]

        return tensordict
