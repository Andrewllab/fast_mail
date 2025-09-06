from __future__ import annotations

from typing import Literal

from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import Transform


class SubsampleTrajectory(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        factor: int,
        split_or_decimate: Literal["split", "decimate"] = "split",
    ):
        self._specs = specs
        self.factor = factor
        self.op = split_or_decimate

    @property
    def specs(self) -> DataSpecs:
        return self._specs
    
    def __call__(self, tensordict: TensorDict) -> list[TensorDict]:
        # This gets called when running with an environment, where the
        # preprocess transforms get rolled into the cpu_batch_transforms.
        # We don't need to do anything, since we can only run during
        # preprocessing.
        return tensordict

    def call_trajectory(self, tensordict: TensorDict) -> list[TensorDict]:

        # We assume that the trajectory has not been prechunked yet, i.e. the
        # subsampling happens relatively early in preprocessing
        for key in ["action", "ref_action"]:
            if tensordict[key].shape[0] != tensordict["obs"].shape[0]:
                raise ValueError(
                    "Action and observation trajectory lengths do not match: "
                    f"{tensordict[key].shape[0]} vs {tensordict['obs'].shape[0]}"
                )

        if self.op == "decimate":
            # we want to keep the very last time step, and rather remove some
            # more initial frames
            offsets = [(tensordict.shape[0] - 1) % self.factor]

        else:
            assert self.op == "split"
            offsets = list(range(self.factor))

        trajs = []
        for offset in offsets:
            # subsample the trajectory
            # TODO: slice anything with a leading dimension of T
            subsampled = TensorDict(
                {
                    "obs": tensordict["obs"][offset :: self.factor],
                    "action": tensordict["action"][offset :: self.factor],
                    "ref_action": tensordict["ref_action"][offset :: self.factor],
                },
            )

            if "goal" in tensordict:
                # don't subsample goals
                subsampled["goal"] = tensordict["goal"]

            trajs.append(subsampled)

        return trajs
