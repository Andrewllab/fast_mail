"""CPU tests for streaming bookkeeping, without downloading foundation models."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import h5py
import numpy as np
import torch

from alex.simulate_static import HDF5FrameReader, read_config
from alex.track_online_pipe import OnlineKeypointTracker, OnlineTrackingPipeline, SAM3Segmenter


class FakeOnlinePredictor:
    """Check window alignment and mutate accumulators just like online CoTracker."""

    def __init__(self):
        self.step = 8
        self.v2 = False
        self.model = SimpleNamespace(window_len=16)
        self.calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, video, is_first_step=False, queries=None, add_support_grid=False):
        if is_first_step:
            self.queries = queries
            self.origin = int(video[0, 0, 0, 0, 0])
            self.model.online_ind = 0
            self.model.online_track_feat = [torch.zeros(1)]
            self.model.online_track_support = [torch.zeros(1)]
            self.model.online_coords_predicted = None
            self.model.online_vis_predicted = None
            self.model.online_conf_predicted = None
            return None, None
        window_start = self.model.online_ind
        if video.shape[1] != 16:
            raise AssertionError("Inference requires exactly 16 frames")
        received = video[0, :, 0, 0, 0].int().tolist()
        expected = list(range(self.origin + window_start, self.origin + window_start + video.shape[1]))
        if received != expected:
            raise AssertionError(f"Incorrect overlapping chunk: {received} != {expected}")
        self.calls.append(received)
        expected_commits = window_start // self.step
        if self.model.online_track_feat[0].item() != expected_commits:
            raise AssertionError("Incorrect online feature state")
        self.model.online_track_feat[0] += 1
        self.model.online_track_support[0] += 1
        length = window_start + video.shape[1]
        tracks = self.queries[..., 1:].unsqueeze(1).repeat(1, length, 1, 1)
        tracks[..., 0] += torch.arange(length).view(1, length, 1)
        visibility = torch.ones(tracks.shape[:-1], dtype=torch.bool)
        self.model.online_coords_predicted = tracks
        self.model.online_vis_predicted = visibility.float()
        self.model.online_conf_predicted = visibility.float()
        self.model.online_ind += self.step
        return tracks, visibility


class FixedMaskSegmenter(SAM3Segmenter):
    def segment(self, images, selections):
        self.initial_images = images
        self.masks = {camera: {"object": np.ones(frame.shape[:2], dtype=bool)} for camera, frame in images.items()}
        return self.masks


class SegmenterTests(unittest.TestCase):
    def test_click_processor_coordinates_mask_selection_and_multiple_cameras(self):
        from transformers import Sam3ImageProcessor, Sam3TrackerProcessor

        processor = Sam3TrackerProcessor(
            image_processor=Sam3ImageProcessor(size={"height": 32, "width": 32})
        )

        def model(**inputs):
            self.assertEqual(tuple(inputs["input_points"].shape), (1, 1, 2, 2))
            self.assertEqual(inputs["input_labels"].tolist(), [[[1, 0]]])
            predictions = torch.full((1, 1, 3, 8, 8), -10.0)
            predictions[:, :, 1, 2:6, 2:6] = 10
            return SimpleNamespace(pred_masks=predictions, iou_scores=torch.tensor([[[0.1, 0.9, 0.2]]]))

        segmenter = SAM3Segmenter(["left", "right"], device="cpu")
        segmenter._models["click"] = (model, processor)
        images = {camera: np.zeros((24, 40, 3), np.uint8) for camera in segmenter.camera_names}
        selections = {camera: [{"name": "cup", "points": [[20, 12], [1, 1]], "labels": [1, 0]}] for camera in images}
        masks = segmenter.segment(images, selections)
        for camera in images:
            self.assertEqual(masks[camera]["cup"].shape, (24, 40))
            self.assertTrue(masks[camera]["cup"][12, 20])
            self.assertFalse(masks[camera]["cup"][0, 0])

    def test_text_instances_sorted_and_kept_separate(self):
        class Inputs(dict):
            def to(self, device):
                return self

        class Processor:
            def __call__(self, **kwargs):
                return Inputs(original_sizes=torch.tensor([[8, 8]]))

            def post_process_instance_segmentation(self, outputs, **kwargs):
                masks = torch.zeros((2, 8, 8), dtype=torch.bool)
                masks[0, :4] = True
                masks[1, 4:] = True
                return [{"scores": torch.tensor([0.6, 0.9]), "masks": masks}]

        segmenter = SAM3Segmenter(["left"], device="cpu")
        segmenter._models["text"] = (lambda **kwargs: None, Processor())
        images = {"left": np.zeros((8, 8, 3), dtype=np.uint8)}
        masks = segmenter.segment(images, {"left": [{"name": "cup", "text": "cup", "instances": "all"}]})
        self.assertEqual(list(masks["left"]), ["cup:0", "cup:1"])
        self.assertTrue(masks["left"]["cup:0"][7, 0])
        masks = segmenter.segment(images, {"left": [{"name": "cup", "text": "cup"}]})
        self.assertEqual(list(masks["left"]), ["cup"])


class OnlineTrackerTests(unittest.TestCase):
    def make_tracker(self, cameras=("left", "right")):
        return OnlineKeypointTracker(cameras, device="cpu", grid_spacing=8, max_points_per_object=4, predictor_factory=FakeOnlinePredictor)

    def frames(self, index):
        return {"left": np.full((32, 40, 3), index, np.uint8),
                "right": np.full((24, 48, 3), index + 30, np.uint8)}

    def test_warmup_eight_frame_updates_and_persistent_ids(self):
        tracker = self.make_tracker()
        frames = self.frames(0)
        masks = {camera: {"cup": np.ones(frame.shape[:2], bool), "bowl": np.ones(frame.shape[:2], bool)} for camera, frame in frames.items()}
        self.assertIsNone(tracker.initialize(frames, masks))
        initial = {camera: predictor.queries[0, :, 1:].numpy().copy() for camera, predictor in tracker.predictors.items()}
        for index in range(1, 42):
            result = tracker.push(self.frames(index))
            if index < 15:
                self.assertIsNone(result)
                self.assertIsNone(tracker.get_latest_keypoints())
                self.assertFalse(tracker.updated)
                self.assertEqual(tracker.inference_count, 0)
                continue
            inferred_index = 15 + ((index - 15) // 8) * 8
            self.assertEqual(tracker.updated, index == inferred_index)
            for camera in frames:
                np.testing.assert_array_equal(result[camera].point_ids, np.arange(8))
                np.testing.assert_array_equal(result[camera].object_names, ["cup"] * 4 + ["bowl"] * 4)
                np.testing.assert_allclose(result[camera].xy, initial[camera] + [inferred_index, 0])
                self.assertEqual(result[camera].frame_index, inferred_index)
                self.assertEqual(result[camera].status, "inferred")
                self.assertLessEqual(len(tracker.buffers[camera]), 15)
                self.assertEqual(len(tracker.predictors[camera].calls), (inferred_index - 15) // 8 + 1)
        returned = tracker.get_latest_keypoints("left")
        returned.xy[:] = -100
        self.assertFalse((tracker.get_latest_keypoints("left").xy == -100).any())
        self.assertEqual(tracker.tracks["left"].shape, (40, 8, 2))
        self.assertIsNot(tracker.predictors["left"], tracker.predictors["right"])
        tracker.finish()
        with self.assertRaises(RuntimeError):
            tracker.push(self.frames(42))
        tracker.reset()
        self.assertIsNone(tracker.initialize(frames, masks))

    def test_single_frame_empty_masks_and_resolution_change(self):
        tracker = self.make_tracker(("left",))
        frames = {"left": self.frames(0)["left"]}
        with self.assertRaises(ValueError):
            tracker.initialize(frames, {"left": {"empty": np.zeros((32, 40), bool)}})
        tracker.initialize(frames, {"left": {"cup": np.ones((32, 40), bool)}})
        with self.assertRaises(ValueError):
            tracker.push({"left": np.zeros((33, 40, 3), np.uint8)})
        with self.assertRaises(ValueError):
            tracker.push({"wrong": frames["left"]})
        self.assertIsNone(tracker.finish())

    def test_direct_named_points_bypass_segmentation(self):
        tracker = self.make_tracker()
        frames = self.frames(0)
        selections = {
            "left": [
                {"name": "hand", "points": [[4, 5], [8, 9]]},
                {"name": "block", "points": [[12, 13]]},
            ],
            "right": [
                {"name": "hand", "points": [[6, 7]]},
                {"name": "block", "points": [[10, 11], [14, 15]]},
            ],
        }
        self.assertIsNone(tracker.initialize_points(frames, selections))
        np.testing.assert_array_equal(
            tracker.identities["left"][1], ["hand", "hand", "block"]
        )
        np.testing.assert_allclose(
            tracker.predictors["right"].queries[0, :, 1:],
            [[6, 7], [10, 11], [14, 15]],
        )

    def test_export_contains_inferred_timeline_without_unprocessed_tail(self):
        tracker = self.make_tracker()
        pipeline = OnlineTrackingPipeline(FixedMaskSegmenter(tracker.camera_names, device="cpu"), tracker)
        pipeline.initialize(self.frames(0), {})
        for index in range(1, 26):
            pipeline.push(self.frames(index))
        pipeline.finish()
        with tempfile.TemporaryDirectory() as directory:
            pipeline.segmenter.visualize(directory)
            pipeline.visualize_current(directory)
            paths = pipeline.export(directory, save_videos=True)
            for camera, artifacts in paths.items():
                with np.load(artifacts["tracks"]) as saved:
                    self.assertEqual(saved["tracks"].shape, (24, 4, 2))
                    np.testing.assert_array_equal(saved["frame_indices"], np.arange(24))
                    np.testing.assert_allclose(saved["tracks"][:, 0, 0], saved["tracks"][0, 0, 0] + np.arange(24))
                    self.assertEqual(saved["inference_ms"].shape, (2,))
                capture = cv2.VideoCapture(artifacts["video"])
                count = 0
                while capture.read()[0]:
                    count += 1
                capture.release()
                self.assertEqual(count, 24)
            self.assertEqual(pipeline.get_latest_keypoints("left").frame_index, 23)
            self.assertEqual(int(pipeline.latest_images["left"][0, 0, 0]), 23)

    def test_history_can_be_disabled(self):
        tracker = self.make_tracker()
        pipeline = OnlineTrackingPipeline(FixedMaskSegmenter(tracker.camera_names, device="cpu"), tracker, keep_history=False)
        pipeline.initialize(self.frames(0), {})
        pipeline.push(self.frames(1))
        self.assertEqual(pipeline.history["left"], [])
        self.assertEqual(pipeline.inference_times_ms, [])
        with self.assertRaises(RuntimeError):
            pipeline.export("unused")


class SourceTests(unittest.TestCase):
    def recording(self, path, right_length=7):
        with h5py.File(path, "w") as handle:
            for camera, length in (("left", 7), ("right", right_length)):
                pixels = np.arange(length, dtype=np.uint8)[:, None, None, None]
                handle.create_dataset(f"obs/{camera}/frames/rgb", data=np.broadcast_to(pixels, (length, 8, 10, 3)))
                handle.create_dataset(f"obs/{camera}/frames/time", data=np.arange(length) / 15)

    def test_rgb_pointer_striding_resize_and_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            self.recording(path)
            with HDF5FrameReader(str(path), start=1, stop=7, stride=2, resize_wh=[6, 4]) as source:
                self.assertEqual(len(source), 3)
                for index, source_index in enumerate((1, 3, 5)):
                    self.assertEqual(source.pointer, index)
                    bundle = source.next_frame()
                    self.assertEqual(source.pointer, index + 1)
                    self.assertEqual((bundle.index, bundle.source_index), (index, source_index))
                    self.assertEqual(bundle.original_sizes_wh["left"], (10, 8))
                    self.assertEqual(bundle.images["left"].shape, (4, 6, 3))
                    self.assertEqual(int(bundle.images["left"][0, 0, 0]), source_index)
                self.assertIsNone(source.next_frame())
                self.assertIsNone(source.next_frame())
                self.assertEqual(source.pointer, 3)
            with self.assertRaises(RuntimeError):
                source.next_frame()

    def test_misaligned_lengths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode.h5"
            self.recording(path, right_length=6)
            with self.assertRaises(ValueError):
                HDF5FrameReader(str(path))

    def test_rgb_conversion_and_float_validation(self):
        image = np.zeros((4, 6, 3), dtype=np.float32)
        image[..., 0] = 1
        rgb = HDF5FrameReader.preprocess_image(image, color_order="BGR", float_range="0_1")
        np.testing.assert_array_equal(rgb[0, 0], [0, 0, 255])
        self.assertTrue(rgb.flags.c_contiguous)
        with self.assertRaises(ValueError):
            HDF5FrameReader.preprocess_image(image * 2, float_range="0_1")

    def test_dictionary_overrides(self):
        config, inspect_only = read_config(["--inspect", "--set", "source.stop=3", "--set", "save_videos=false", "--set", 'source.camera_paths={"left":"obs/left/frames/rgb"}'])
        self.assertTrue(inspect_only)
        self.assertEqual(config["source"]["stop"], 3)
        self.assertFalse(config["save_videos"])
        self.assertEqual(config["source"]["camera_paths"], {"left": "obs/left/frames/rgb"})


if __name__ == "__main__":
    unittest.main()
