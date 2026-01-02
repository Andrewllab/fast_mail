from __future__ import annotations

from typing import List, Optional

import torch
from omegaconf import ListConfig
from PIL import Image
from tensordict import TensorDict
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
)

from environments.specs import DataSpecs
from transforms.base_transform import Transform


def _gdino_text(text: str) -> str:
    # GroundingDINO HF usage: lowercase + end with a dot :contentReference[oaicite:2]{index=2}
    t = (text or "").strip().lower()
    if not t.endswith("."):
        t += "."
    return t


class SamV3SegmenterTransform(Transform):
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
        segmentation_text_keys: Optional[str | List[str]] = None,
        segmentation_text_prompts: Optional[str | List[str]] = None,
        segmentation_text_source: str = "prompt",
        camera_keys: str | List[str] = "left_cam",
        device: str = "cuda",
        # GroundingDINO
        gdino_model_id: str = "IDEA-Research/grounding-dino-base",
        gdino_box_threshold: float = 0.3,
        gdino_text_threshold: float = 0.3,
        # Anchor + tracking
        anchor_frame_idx: int = 0,
        sam_dtype: torch.dtype = torch.bfloat16,
        max_frame_num_to_track: int = 10_000,
    ):
        super().__init__()
        self._specs = specs
        self.device = torch.device(device)

        # ---- GroundingDINO (box proposals) :contentReference[oaicite:4]{index=4}
        self.gdino_processor = AutoProcessor.from_pretrained(gdino_model_id)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            gdino_model_id
        ).to(self.device)
        self.gdino_model.eval()
        self.gdino_box_threshold = gdino_box_threshold
        self.gdino_text_threshold = gdino_text_threshold

        # ---- SAM3 Tracker Video (instance tracking from visual prompts) :contentReference[oaicite:5]{index=5}
        self.sam_model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3").to(
            self.device, dtype=sam_dtype
        )
        self.sam_processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")

        self.anchor_frame_idx = anchor_frame_idx
        self.max_frame_num_to_track = max_frame_num_to_track

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
            raise ValueError("segmentation_text_source must be 'prompt' or 'key'")

        self.segmentation_text_source = segmentation_text_source

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    @torch.no_grad()
    def _gdino_best_box_xyxy(
        self, frame: Image.Image, text_prompt: str
    ) -> Optional[torch.Tensor]:
        """
        Returns best detection box as torch.Tensor[4] in XYXY absolute pixels, on self.device.
        """
        text = _gdino_text(text_prompt)

        inputs = self.gdino_processor(images=frame, text=text, return_tensors="pt").to(
            self.device
        )
        outputs = self.gdino_model(**inputs)

        # target_sizes expects [height,width]
        H, W = frame.size[1], frame.size[0]
        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.gdino_box_threshold,
            text_threshold=self.gdino_text_threshold,
            target_sizes=[(H, W)],
        )
        r0 = results[0]
        if r0["boxes"].numel() == 0:
            return None

        best_i = int(torch.argmax(r0["scores"]).item())
        return r0["boxes"][best_i].to(self.device)  # [x1,y1,x2,y2]

    @torch.no_grad()
    def _gdino_best_box_xyxy_video(
        self, frames: List[Image.Image], text_prompt: str
    ) -> Optional[torch.Tensor]:
        """
        Returns best detection box as torch.Tensor[4] in XYXY absolute pixels, on self.device.
        """
        text = _gdino_text(text_prompt)

        best_frame_idx = -1
        best_score = -float("inf")
        best_box = None

        for frame_idx, frame in enumerate(frames):
            inputs = self.gdino_processor(
                images=frame, text=text, return_tensors="pt"
            ).to(self.device)
            outputs = self.gdino_model(**inputs)

            # target_sizes expects [height,width]
            H, W = frame.shape[0], frame.shape[1]
            results = self.gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.gdino_box_threshold,
                text_threshold=self.gdino_text_threshold,
                target_sizes=[(H, W)],
            )
            r0 = results[0]
            if r0["boxes"].numel() == 0:
                continue

            best_i = int(torch.argmax(r0["scores"]).item())
            if r0["scores"][best_i] > best_score:
                best_score = r0["scores"][best_i]
                best_box = r0["boxes"][best_i]
                best_frame_idx = frame_idx
        if best_box is None:
            return None, -1
        return best_box.to(self.device), best_frame_idx  # [x1,y1,x2,y2]

    @torch.no_grad()
    def _track_with_sam3_tracker(
        self,
        video_frames: List[Image.Image],
        ann_frame_idx: int,
        init_box_xyxy: torch.Tensor,
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
        box = init_box_xyxy.tolist()
        # Shape conventions match SAM2/SAM3 video processors: [image][box][coords]
        self.sam_processor.add_inputs_to_inference_session(
            inference_session=inference_session,
            frame_idx=ann_frame_idx,
            obj_ids=obj_id,
            input_boxes=[[box]],
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
            video = tensordict["obs"][camera_key]["rgb"]  # [T,H,W,C]
            T = int(video.shape[0])

            if self.segmentation_text_source == "prompt":
                segmentation_texts = self.segmentation_text_prompts
            else:
                segmentation_texts = [
                    tensordict["goal"][k] for k in self.segmentation_text_keys
                ]

            # Convert whole video to PIL once (SAM tracker wants list of frames) :contentReference[oaicite:7]{index=7}
            video_frames = video
            ann_frame_idx = int(max(0, min(T - 1, self.anchor_frame_idx)))

            for text_prompt, segmenter_out_key in zip(
                segmentation_texts, self.segmenter_out_key
            ):
                # 1) GroundingDINO on anchor frame to pick initial object :contentReference[oaicite:8]{index=8}
                init_box, ann_frame_idx = self._gdino_best_box_xyxy_video(
                    video_frames, str(text_prompt)
                )

                # If no detection, return empty mask
                H, W = video.shape[1], video.shape[2]
                segmentations = torch.zeros(
                    (T, H, W), device=self.device, dtype=torch.bool
                )
                if init_box is None:
                    tensordict["obs", camera_key, segmenter_out_key] = segmentations
                    continue

                # 2) Track forward from anchor frame
                fwd_masks = self._track_with_sam3_tracker(
                    video_frames=video_frames,
                    ann_frame_idx=ann_frame_idx,
                    init_box_xyxy=init_box,
                )

                # 3) Track backward by running on reversed video from the corresponding anchor
                # rev_frames = torch.flip(video_frames, dims=[0])
                # rev_ann_idx = (T - 1) - ann_frame_idx

                bwd_masks = self._track_with_sam3_tracker(
                    video_frames=video_frames,
                    ann_frame_idx=ann_frame_idx,
                    init_box_xyxy=init_box,
                    reverse=True,
                )

                # 4) Merge (OR). If you prefer “trust forward over backward”, replace this with a directional stitch.
                segmentations = fwd_masks | bwd_masks

                tensordict["obs", camera_key, segmenter_out_key] = segmentations

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        return tensordict
