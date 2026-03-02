import dataclasses
from typing import Optional

import torch
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import Transform


class ApplyRobocasaSegmentations(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        camera_keys: list[str],
        mask_out_keys: list[str],
        segmentation_keys: Optional[list[str] | str] = None,
        segmentation_goal_keys: Optional[list[str] | str] = None,
    ):

        if segmentation_keys is not None and segmentation_goal_keys is not None:
            raise ValueError(
                "Only one of segmentation_keys and segmentation_goal_keys can be provided"
            )

        self.camera_keys = camera_keys
        if isinstance(segmentation_keys, str):
            segmentation_keys = [segmentation_keys]
        self.segmentation_keys = segmentation_keys
        if isinstance(segmentation_goal_keys, str):
            segmentation_goal_keys = [segmentation_goal_keys]
        self.segmentation_goal_keys = segmentation_goal_keys
        if isinstance(mask_out_keys, str):
            mask_out_keys = [mask_out_keys]
        self.mask_out_keys = mask_out_keys
        if isinstance(mask_out_keys, str):
            mask_out_keys = [mask_out_keys]

        if self.segmentation_keys is not None and len(self.segmentation_keys) != len(
            self.mask_out_keys
        ):
            raise ValueError(
                "segmentation_keys and mask_out_keys must have the same length"
            )
        if self.segmentation_goal_keys is not None and len(
            self.segmentation_goal_keys
        ) != len(self.mask_out_keys):
            raise ValueError(
                "segmentation_goal_keys and mask_out_keys must have the same length"
            )

        self._input_specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._input_specs

    def call_trajectory(self, tensordict):
        segmentation_ids = tensordict["segmentation_ids"]

        segmentation_keys = self.segmentation_keys
        if segmentation_keys is None:
            segmentation_keys = [
                tensordict["goal"][key] for key in self.segmentation_goal_keys
            ]

        for cam_key in self.camera_keys:
            segmentation = tensordict[("obs", cam_key, "segmentation")]

            for seg_key, mask_out_key in zip(segmentation_keys, self.mask_out_keys):
                local_seg_ids = torch.tensor(
                    segmentation_ids[seg_key], device=segmentation.device
                )
                mask = torch.isin(segmentation, local_seg_ids)
                tensordict[("obs", cam_key, mask_out_key)] = mask

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        if "segmentation_ids" not in tensordict:
            segmentation_ids = tensordict["obs"]["segmentation_ids"]
        else:
            segmentation_ids = tensordict["segmentation_ids"]

        segmentation_keys = self.segmentation_keys
        if segmentation_keys is None:
            segmentation_keys = tensordict["goal"][self.segmentation_goal_keys]

        for cam_key in self.camera_keys:
            segmentation = tensordict[("obs", cam_key, "segmentation_element")]

            for seg_key, mask_out_key in zip(
                self.segmentation_keys, self.mask_out_keys
            ):
                local_seg_ids = torch.tensor(
                    segmentation_ids[seg_key], device=segmentation.device
                )
                mask = (
                    segmentation.unsqueeze(-1)
                    == local_seg_ids.unsqueeze(1).unsqueeze(1)
                ).any(
                    dim=-1
                )  # torch.isin(segmentation, local_seg_ids)
                tensordict[("obs", cam_key, mask_out_key)] = mask

        return tensordict
