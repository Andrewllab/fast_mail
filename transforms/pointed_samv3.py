from __future__ import annotations

from typing import List, Optional

import torch
from omegaconf import ListConfig
from PIL import Image
from tensordict import TensorDict
from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

from environments.specs import DataSpecs
from transforms.base_transform import Transform


class SamV3VideoSegmenterTransform(Transform):
    """
    GroundingDINO selects initial object (box) on an anchor frame,
    then SAM3 Tracker Video propagates the mask through the whole video,
    both forward and backward (by also running on reversed video).

    Notes:
      - Output remains: tensordict["obs", camera_key, segmenter_out_key] = [T,H,W] bool
      - Uses SAM3 Tracker Video (PVS), not SAM3 Video PCS. :contentReference[oaicite:3]{index=3}
    """

    def __init__(
        self,
        specs: DataSpecs,
        segmenter_out_keys: str | List[str],
        segmentation_click_keys: Optional[str | List[str]] = None,
        camera_keys: str | List[str] = "left_cam",
        rgb_key: str = "rgb",
        device: str = "cuda",
        out_device: str | torch.device = "cpu",
        # Anchor + tracking
        sam_dtype: torch.dtype = torch.bfloat16,
        max_frame_num_to_track: int = 10_000,
        add_backward_tracking: bool = True,
    ):
        super().__init__()
        self._specs = specs
        self.device = torch.device(device)
        self.out_device = torch.device(out_device)
        # ---- SAM3 Tracker Video (instance tracking from visual prompts) :contentReference[oaicite:5]{index=5}
        self.sam_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(
            self.device, dtype=sam_dtype
        )
        self.sam_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

        self.max_frame_num_to_track = max_frame_num_to_track

        self.camera_keys = (
            camera_keys
            if isinstance(camera_keys, (list, ListConfig))
            else [camera_keys]
        )
        self.segmenter_out_keys = (
            segmenter_out_keys
            if isinstance(segmenter_out_keys, (list, ListConfig))
            else [segmenter_out_keys]
        )

        self.segmentation_click_keys = (
            segmentation_click_keys
            if isinstance(segmentation_click_keys, (list, ListConfig))
            else [segmentation_click_keys]
        )

        if len(self.segmenter_out_keys) != len(self.segmentation_click_keys):
            raise ValueError(
                "segmenter_out_keys length must match segmentation_click_keys length"
            )

        self.rgb_key = rgb_key
        self.add_backward_tracking = add_backward_tracking

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    @torch.no_grad()
    def _track_with_sam3_tracker(
        self,
        video_frames: List[Image.Image],
        ann_frame_idx: int,
        init_points: List[List[float]],
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Returns masks as bool tensor [T,H,W] on self.device.
        """
        T = len(video_frames)
        H, W = video_frames[0].shape[0], video_frames[0].shape[1]

        inference_session = self.sam_processor.init_video_session(
            processing_device="cpu",
            video_storage_device="cpu",
            video=video_frames,
            inference_device=self.device,
            dtype=self.sam_model.dtype,
        )

        obj_id = 1

        # Prefer box prompt (more deterministic). If processor/model raises, fallback to center-point click.
        # Shape conventions match SAM2/SAM3 video processors: [image][box][coords]
        self.sam_processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=obj_id,
            input_points=[[init_points]],
            input_labels=[[[1] * len(init_points)]],  # 1 for foreground
        )

        masks = torch.zeros((T, H, W), dtype=torch.bool, device=self.device)

        # Propagate forward from the session’s start; output.frame_idx corresponds to video index
        for out in self.sam_model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=self.max_frame_num_to_track,
            start_frame_idx=ann_frame_idx,
            reverse=reverse,
        ):
            # out.pred_masks -> resize/binarize at original resolution
            video_res_masks = self.sam_processor.post_process_masks(
                [out.pred_masks],
                original_sizes=[
                    [inference_session.video_height, inference_session.video_width]
                ],
                binarize=True,
            )[
                0
            ]  # [num_obj, 1, H, W] or [num_obj, H, W] depending on version

            m = video_res_masks
            if m.ndim == 4:
                m = m[:, 0]  # [num_obj,H,W]
            masks[out.frame_idx] = m[0] > 0

        inference_session.reset_inference_session()
        return masks

    @torch.no_grad()
    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        for camera_key in self.camera_keys:
            video = tensordict["obs"][camera_key][self.rgb_key]  # [T,H,W,C]
            T = int(video.shape[0])

            # Convert whole video to PIL once (SAM tracker wants list of frames) :contentReference[oaicite:7]{index=7}
            video_frames = video
            # ann_frame_idx = int(max(0, min(T - 1, self.anchor_frame_idx)))

            for segmentation_click_key, segmenter_out_key in zip(
                self.segmentation_click_keys, self.segmenter_out_keys
            ):
                # If no detection, return empty mask
                H, W = video.shape[1], video.shape[2]
                segmentations = torch.zeros(
                    (T, H, W), device=self.device, dtype=torch.bool
                )

                segmentation_clicks = tensordict["obs", camera_key].get(
                    segmentation_click_key, None
                )
                if segmentation_clicks is None:
                    tensordict["obs", camera_key, segmenter_out_key] = segmentations.to(
                        self.out_device
                    )
                    continue

                annotation_timestep = int(segmentation_clicks[0].data["timestep"])
                annotation_points = []
                for click in segmentation_clicks:
                    if int(click.data["timestep"]) != annotation_timestep:
                        raise ValueError(
                            "All clicks for a given camera and object must be on the same timestep"
                        )
                    annotation_points.append([click.data["x"], click.data["y"]])

                # 2) Track forward from anchor frame
                segmentations = self._track_with_sam3_tracker(
                    video_frames=video_frames,
                    ann_frame_idx=annotation_timestep,
                    init_points=annotation_points,
                )

                # 3) Track backward by running on reversed video from the corresponding anchor
                # rev_frames = torch.flip(video_frames, dims=[0])
                # rev_ann_idx = (T - 1) - ann_frame_idx

                if self.add_backward_tracking:
                    bwd_masks = self._track_with_sam3_tracker(
                        video_frames=video_frames,
                        ann_frame_idx=annotation_timestep,
                        init_points=annotation_points,
                        reverse=True,
                    )

                    # 4) Merge (OR). If you prefer “trust forward over backward”, replace this with a directional stitch.
                    segmentations = segmentations | bwd_masks

                tensordict["obs", camera_key, segmenter_out_key] = segmentations.to(
                    self.out_device
                )

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        return tensordict
