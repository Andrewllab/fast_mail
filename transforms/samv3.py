from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import requests
import torch
from omegaconf import ListConfig
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
        segmenter_out_keys: str | List[str],
        segmentation_text_keys: Optional[str | List[str]] = None,
        segmentation_text_prompts: Optional[str | List[str]] = None,
        segmentation_text_source: str = "prompt",
        camera_keys: str | List[str] = "left_cam",
        device: str | torch.device = "cuda",
    ):
        super().__init__()

        self.model = Sam3VideoModel.from_pretrained("facebook/sam3").to(
            device,
            dtype=torch.bfloat16,
        )
        self.processor = Sam3VideoProcessor.from_pretrained("facebook/sam3")

        self._specs = specs
        self.device = device

        # General settings
        self.camera_keys = (
            camera_keys
            if isinstance(camera_keys, (list, ListConfig))
            else [camera_keys]
        )
        self.segmenter_out_key = (
            segmenter_out_keys
            if isinstance(segmenter_out_keys, (list, ListConfig))
            else [segmenter_out_keys]
        )
        if segmentation_text_source == "prompt":
            self.segmentation_text_prompts = (
                segmentation_text_prompts
                if isinstance(segmentation_text_prompts, (list, ListConfig))
                else [segmentation_text_prompts]
            )
        elif segmentation_text_source == "key":
            self.segmentation_text_keys = (
                segmentation_text_keys
                if isinstance(segmentation_text_keys, (list, ListConfig))
                else [segmentation_text_keys]
            )
        else:
            raise ValueError(
                f"Invalid segmentation_text_source: {segmentation_text_source}, must be 'prompt' or 'key'!"
            )

        self.segmentation_text_source = segmentation_text_source

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    @torch.no_grad()
    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        for camera_key in self.camera_keys:
            video = tensordict["obs"][camera_key]["rgb"]

            if self.segmentation_text_source == "prompt":
                segmentation_texts = self.segmentation_text_prompts
            elif self.segmentation_text_source == "key":
                segmentation_texts = [
                    tensordict["goal"][text_key]
                    for text_key in self.segmentation_text_keys
                ]

            for text_prompt, segmenter_out_key in zip(
                segmentation_texts, self.segmenter_out_key
            ):
                # Initialize video inference session
                inference_session = self.processor.init_video_session(
                    video=video,
                    inference_device=self.device,
                    processing_device="cpu",
                    video_storage_device="cpu",
                    dtype=torch.bfloat16,
                )

                segmentations = torch.zeros(
                    *video.shape[:3], device=self.device, dtype=torch.bool
                )

                # Add text prompt
                inference_session = self.processor.add_text_prompt(
                    inference_session=inference_session,
                    text=text_prompt,
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
                obj_scores = {}
                for outputs in outputs_per_frame.values():
                    for obj_id, obj_conf in zip(
                        outputs["object_ids"], outputs["scores"]
                    ):
                        obj_scores[obj_id.item()] = obj_conf.item()

                if not obj_scores:
                    tensordict["obs", camera_key, segmenter_out_key] = segmentations
                    continue
                max_score_obj_id = max(obj_scores, key=obj_scores.get)

                for frame_idx in range(video.shape[0]):
                    if outputs_per_frame[frame_idx]["masks"].numel() == 0:
                        continue

                    for obj_idx, obj_id in enumerate(
                        outputs_per_frame[frame_idx]["object_ids"]
                    ):
                        if obj_id.item() == max_score_obj_id:
                            segmentations[frame_idx] = outputs_per_frame[frame_idx][
                                "masks"
                            ][obj_idx]
                            break
                    # segmentations[frame_idx] = (
                    #     outputs_per_frame[frame_idx]["masks"].sum(dim=0) > 0
                    # )  # Combine masks for the same object id

                tensordict["obs", camera_key, segmenter_out_key] = segmentations

                # Reset inference session
                inference_session.reset_inference_session()

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Do nothing if not called during preprocessing
        return tensordict
