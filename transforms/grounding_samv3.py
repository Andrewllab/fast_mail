from __future__ import annotations

from typing import List, Optional

import torch
from omegaconf import ListConfig
from PIL import Image
from tensordict import TensorDict
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    Sam3Model,
    Sam3Processor,
    Sam3TrackerVideoModel,
    Sam3TrackerVideoProcessor,
)

from environments.specs import DataSpecs
from transforms.base_transform import Transform


def _gdino_text(text: str) -> str:
    # GroundingDINO: lowercase + end with dot.
    t = (text or "").strip().lower()
    if not t.endswith("."):
        t += "."
    return t


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


class SamV3PictureSegmenterTransform(Transform):
    """
    Per-frame pipeline:
      1) GroundingDINO detects boxes on each frame (batched).
      2) SAM3 Tracker (image) segments each frame independently using the box prompt (batched).
    No video session, no tracking IDs, no propagation.

    Output:
      tensordict["obs", camera_key, segmenter_out_key] = torch.bool [T, H, W]
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
        # GroundingDINO
        gdino_model_id: str = "IDEA-Research/grounding-dino-base",
        gdino_box_threshold: float = 0.3,
        gdino_text_threshold: float = 0.3,
        # Batching
        max_frames_per_batch: int = 16,
        # SAM3 Tracker image settings
        sam_dtype: torch.dtype = torch.bfloat16,
        # Runtime improvements
        compile_sam: bool = False,
        compile_gdino: bool = False,
    ):
        super().__init__()
        self._specs = specs
        self.device = torch.device(device) if isinstance(device, str) else device

        # GroundingDINO :contentReference[oaicite:4]{index=4}
        self.gdino_processor = AutoProcessor.from_pretrained(gdino_model_id)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            gdino_model_id
        ).to(self.device)
        self.gdino_model.eval()
        self.gdino_box_threshold = gdino_box_threshold
        self.gdino_text_threshold = gdino_text_threshold

        if compile_gdino:
            self.gdino_model = torch.compile(self.gdino_model, mode="reduce-overhead")

        # SAM3 Tracker (single-image promptable visual segmentation) :contentReference[oaicite:5]{index=5}
        self.sam_processor = Sam3Processor.from_pretrained("facebook/sam3")
        self.sam_model = Sam3Model.from_pretrained("facebook/sam3").to(
            self.device, dtype=sam_dtype
        )

        self.sam_model.eval()

        if compile_sam:
            self.sam_model = torch.compile(self.sam_model, mode="reduce-overhead")

        self.max_frames_per_batch = max_frames_per_batch

        # Same pattern as your existing transform :contentReference[oaicite:6]{index=6}
        self.camera_keys = camera_keys
        if isinstance(self.camera_keys, str):
            self.camera_keys = [self.camera_keys]
        self.segmenter_out_key = segmenter_out_keys
        if isinstance(self.segmenter_out_key, str):
            self.segmenter_out_key = [self.segmenter_out_key]
        if len(self.segmenter_out_key) != len(self.camera_keys):
            raise ValueError("segmenter_out_keys length must match camera_keys length")

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
        self.sam_dtype = sam_dtype

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    @torch.no_grad()
    def _gdino_boxes_for_video(
        self,
        video_bchw_uint8: torch.Tensor,  # [T,3,H,W] uint8 on CPU or GPU
        text_prompt: str,
    ) -> torch.Tensor:
        """
        Returns per-frame best box in XYXY absolute pixels:
          boxes_xyxy: [T,4] (float), with NaNs for frames with no detection.
        """
        T, _, H, W = video_bchw_uint8.shape
        text = _gdino_text(text_prompt)

        boxes_xyxy = torch.full((T, 4), float("nan"))
        box_scores = torch.full((T,), float("-inf"))

        # Batch the frames to keep memory stable.
        for start in range(0, T, self.max_frames_per_batch):
            end = min(T, start + self.max_frames_per_batch)
            frames = video_bchw_uint8[start:end]  # [B,3,H,W]

            inputs = self.gdino_processor(
                images=frames,  # torch tensor batch
                text=[text]
                * (end - start),  # same text per frame to avoid batch-text edge cases
                return_tensors="pt",
            ).to(self.device)

            outputs = self.gdino_model(**inputs)

            # target_sizes expects list/array of (height,width) per image
            target_sizes = [(H, W)] * (end - start)
            results = self.gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.gdino_box_threshold,
                text_threshold=self.gdino_text_threshold,
                target_sizes=target_sizes,
            )

            for bi, r in enumerate(results):
                if r["boxes"].numel() == 0:
                    continue
                best_i = int(torch.argmax(r["scores"]).item())
                boxes_xyxy[start + bi] = r["boxes"][best_i]
                box_scores[start + bi] = r["scores"][best_i]

        return boxes_xyxy, box_scores

    @torch.no_grad()
    def _sam_masks_from_boxes(
        self,
        video_bchw_uint8: torch.Tensor,  # [T,3,H,W] uint8
        boxes_xyxy: torch.Tensor,  # [T,4] float (xyxy), may contain NaNs
    ) -> torch.Tensor:
        """
        Uses SAM3 Tracker (image) to segment each frame independently with a box prompt.
        Returns masks: [T,H,W] bool on self.device.
        """
        T, _, H, W = video_bchw_uint8.shape
        masks_out = torch.zeros((T, H, W), dtype=torch.bool, device=self.device)
        masks_scores = torch.full((T,), float("-inf"), device=self.device)

        valid = torch.isfinite(boxes_xyxy).all(dim=1)  # [T]
        if not valid.any():
            return masks_out, masks_scores

        # Process only valid frames, but keep output aligned with T
        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        frames_valid = video_bchw_uint8[valid_idx]  # [Tv,3,H,W]
        boxes_valid = boxes_xyxy[valid_idx].unsqueeze(
            1
        )  # [Tv,1,4]  (batch, num_boxes, 4) :contentReference[oaicite:8]{index=8}

        # Batch in chunks
        Tv = frames_valid.shape[0]
        for start in range(0, Tv, self.max_frames_per_batch):
            end = min(Tv, start + self.max_frames_per_batch)
            frames = frames_valid[start:end]
            boxes = boxes_valid[start:end]

            inputs = self.sam_processor(
                images=frames,  # torch tensor batch
                input_boxes=boxes.tolist(),  # [B,1,4] in xyxy pixel coords
                input_boxes_labels=[[1]] * (end - start),  # all foreground
                return_tensors="pt",
            ).to(self.device, dtype=self.sam_dtype)

            outputs = self.sam_model(**inputs)

            # Post-process back to original sizes
            # Returns a list (len=B) of tensors resized to original resolution.
            results_list = self.sam_processor.post_process_instance_segmentation(
                outputs, target_sizes=inputs["original_sizes"].tolist()
            )

            masks_list = [m["masks"] for m in results_list]  # get only masks
            scores_list = [m["scores"] for m in results_list]  # get only scores

            for bi, (mask, scores) in enumerate(zip(masks_list, scores_list)):
                best_score_idx = int(torch.argmax(scores).item())
                m2 = mask[best_score_idx]  # [H,W] float mask

                # Typically float/bool-ish; binarize
                mask_bool = (m2 > 0).to(torch.bool).to(self.device)

                orig_t = int(valid_idx[start + bi].item())
                masks_out[orig_t] = mask_bool
                masks_scores[orig_t] = scores[best_score_idx]

        return masks_out, masks_scores

    @torch.no_grad()
    def call_trajectory(self, tensordict: TensorDict) -> TensorDict:
        for camera_key in self.camera_keys:
            video = tensordict["obs"][camera_key]["rgb"]  # expected [T,H,W,C]
            T, H, W, C = video.shape
            assert C == 3, f"Expected RGB video with 3 channels, got {C}"

            if self.segmentation_text_source == "prompt":
                segmentation_texts = self.segmentation_text_prompts
            else:
                segmentation_texts = [
                    tensordict["goal"][k] for k in self.segmentation_text_keys
                ]

            # Convert to BCHW torch tensor (stay in torch for batching)
            # Keep as uint8; processors will handle scaling/normalization.
            video_bchw = video.permute(0, 3, 1, 2).contiguous()

            for text_prompt, segmenter_out_key in zip(
                segmentation_texts, self.segmenter_out_key
            ):
                # 1) GroundingDINO per-frame boxes (batched)
                boxes_xyxy, box_scores = self._gdino_boxes_for_video(
                    video_bchw, str(text_prompt)
                )

                # 2) SAM3 per-frame segmentation from boxes (batched)
                segmentations, segmentation_scores = self._sam_masks_from_boxes(
                    video_bchw, boxes_xyxy
                )

                tensordict["obs", camera_key, segmenter_out_key] = segmentations
                tensordict["obs", camera_key, f"{segmenter_out_key}_box_xyxy"] = (
                    boxes_xyxy
                )
                tensordict["obs", camera_key, f"{segmenter_out_key}_box_scores"] = (
                    box_scores
                )
                tensordict[
                    "obs", camera_key, f"{segmenter_out_key}_segmentation_scores"
                ] = segmentation_scores

        return tensordict

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        # Do nothing if not called during preprocessing
        return self.call_trajectory(tensordict[0]).unsqueeze(0)
