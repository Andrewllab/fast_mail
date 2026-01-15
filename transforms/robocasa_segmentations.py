import dataclasses

import torch
from tensordict import TensorDict

from environments.specs import DataSpecs
from transforms.base_transform import Transform


class ApplyRobocasaSegmentations(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        camera_keys: list[str],
        segmentation_keys: list[str],
        mask_out_keys: list[str],
    ):

        self.camera_keys = camera_keys
        if isinstance(segmentation_keys, str):
            segmentation_keys = [segmentation_keys]
        self.segmentation_keys = segmentation_keys
        if isinstance(mask_out_keys, str):
            mask_out_keys = [mask_out_keys]
        self.mask_out_keys = mask_out_keys
        if isinstance(mask_out_keys, str):
            mask_out_keys = [mask_out_keys]

        if len(self.segmentation_keys) != len(self.mask_out_keys):
            raise ValueError(
                "segmentation_keys and mask_out_keys must have the same length"
            )

        self._input_specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._input_specs

    def call_trajectory(self, tensordict):
        for cam_key in self.camera_keys:
            segmentation = tensordict[("obs", cam_key, "segmentation")]
            segmentation_ids = tensordict["segmentation_ids"]

            for seg_key, mask_out_key in zip(
                self.segmentation_keys, self.mask_out_keys
            ):
                local_seg_ids = torch.tensor(
                    segmentation_ids[seg_key], device=segmentation.device
                )
                mask = torch.isin(segmentation, local_seg_ids)
                tensordict[("obs", cam_key, mask_out_key)] = mask

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        return self.call_trajectory(tensordict)
