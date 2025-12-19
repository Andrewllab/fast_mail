from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from sklearn.decomposition import PCA
from tensordict import TensorDict
from torch_geometric.data import Data
from transformers import (
    AutoImageProcessor,
    AutoModel,
    Sam3VideoModel,
    Sam3VideoProcessor,
    pipeline,
)
from transformers.video_utils import load_video

from environments.specs import DataSpecs
from transforms.base_transform import Transform


class SamV3SegmenterTransform(Transform):
    """Apply a random translation and rotation to the entire pointcloud
    trajectory during preprocessing, which is then constant throughout
    training.

    Args:
        specs (DataSpecs): The data specifications.
    """

    def __init__(
        self,
        specs: DataSpecs,
        device: str | torch.device = "cuda",
        camera_key: str = "left_cam",
        segmenter_out_key: str = "samv3_segmentation",
        segmentation_text_prompt: str = None,
        feature_out_key: str = "dinov3_features",
    ):
        super().__init__()

        self.model = Sam3VideoModel.from_pretrained("facebook/sam3").to(
            device, dtype=torch.bfloat16
        )
        self.processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

        self._specs = specs
        self.device = device

        # General settings
        self.camera_key = camera_key
        self.segmenter_out_key = segmenter_out_key
        self.segmentation_text_prompt = segmentation_text_prompt
        self.feature_out_key = feature_out_key

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        video = tensordict["obs"][self.camera_key]["rgb"].numpy()

        # Initialize video inference session
        inference_session = self.processor.init_video_session(
            video=video,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.bfloat16,
        )

        # Add text prompt
        inference_session = self.processor.add_text_prompt(
            inference_session=inference_session,
            text=tensordict["goal", "obj_name"],
        )

        # Propagate through video
        outputs_per_frame = {}
        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=inference_session, max_frame_num_to_track=1000
        ):
            processed_outputs = self.processor.postprocess_outputs(
                inference_session, model_outputs
            )
            outputs_per_frame[model_outputs.frame_idx] = processed_outputs

        # Collect segmentations
        segmentations = []
        for frame_idx in range(video.shape[0]):
            segmentation = outputs_per_frame[frame_idx]["masks"].sum(axis=0) > 0
            segmentations.append(torch.tensor(segmentation, device=self.device))

        tensordict["obs", self.camera_key, self.segmenter_out_key] = torch.stack(
            segmentations
        )

        # Reset inference session
        inference_session.reset_inference_session()

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Do nothing if not called during preprocessing
        return tensordict
