"""Hardware-free tests for the RGB-D streaming/recording infrastructure."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import os

import h5py
import numpy as np
import torch

from alex_3d.cameras import BaseRGBDCamera, RGBDFrame
from alex_3d.hdf5_io import (
    RGBDRecording,
    TrackedRGBDRecording,
    save_recording,
    write_tracking_group,
)
from alex_3d.live_utils import (
    FrequencyGate,
    output_frame_indices,
    selected_visualization_fps,
    save_keypoint_overlay,
    visualization_fps,
    write_keypoint_video,
)
from alex_3d.process_recording_offline import (
    offline_sample_indices,
    track_video_in_windows,
)
from alex_3d.selection import flatten_direct_points, sample_mask_grid
from alex_3d.track_online_pipe import CameraCalibration, TrackFrame, lift_keypoints_3d
from alex_3d.terminal_input import TerminalKeyReader


class FakeCamera(BaseRGBDCamera):
    def __init__(self):
        super().__init__("fake")
        self.index = -1

    def initialize(self):
        self.initialized = True
        self.calibration = CameraCalibration(
            np.array([[2.0, 0, 1.0], [0, 2.0, 1.0], [0, 0, 1.0]]),
            np.eye(4), orthographic=True,
        )
        return self.calibration

    def read(self):
        self.index += 1
        return RGBDFrame(
            np.zeros((4, 4, 3), np.uint8), np.full((4, 4), 2.0, np.float32),
            self.calibration.intrinsics, np.eye(4), self.index, float(self.index),
        )

    def close(self):
        self.initialized = False


class StreamingTests(unittest.TestCase):
    def test_camera_and_lifting(self):
        camera = FakeCamera()
        calibration = camera.initialize()
        frame = camera.read()
        points, valid = lift_keypoints_3d(
            calibration, np.array([[1.0, 1.0], [3.0, 3.0]]),
            np.array([True, True]), frame.depth,
        )
        np.testing.assert_allclose(points, [[0, 0, 2], [2, 2, 2]])
        self.assertTrue(valid.all())

    def test_frequency_gate(self):
        gate = FrequencyGate(8)
        gate.reset(0.0)
        self.assertFalse(gate.due(0.124))
        self.assertTrue(gate.due(0.125))
        self.assertFalse(gate.due(0.249))
        self.assertTrue(gate.due(0.250))

    def test_offline_sample_indices_match_online_cadence(self):
        np.testing.assert_array_equal(offline_sample_indices(40), [15, 23, 31, 39])
        np.testing.assert_array_equal(output_frame_indices(5, 0, 1), [0, 1, 2, 3, 4])

    def test_chunked_offline_tracking_preserves_ids_and_continuity(self):
        class FakeOfflinePredictor:
            def __init__(self):
                self.window_lengths = []

            def __call__(self, video, queries, grid_size):
                length = video.shape[1]
                self.window_lengths.append(length)
                points = queries[:, :, 1:]
                query_times = queries[:, :, 0].view(1, 1, -1, 1)
                offsets = torch.arange(length, dtype=points.dtype).view(1, length, 1, 1)
                tracks = points[:, None].repeat(1, length, 1, 1)
                tracks[..., 0:1] += offsets - query_times
                visibility = torch.ones(tracks.shape[:-1], dtype=torch.bool)
                return tracks, visibility

        predictor = FakeOfflinePredictor()
        frames = np.zeros((45, 8, 8, 3), np.uint8)
        points = np.array([[1, 2], [3, 4]], np.float32)
        windows = []
        tracks, visibility = track_video_in_windows(
            predictor, frames, points, torch.device("cpu"),
            max_window_frames=20, overlap_frames=5,
            progress=lambda number, start, end, total: windows.append((start, end)),
        )
        self.assertEqual(windows, [(0, 20), (15, 35), (29, 45)])
        self.assertLessEqual(max(predictor.window_lengths), 20)
        np.testing.assert_allclose(tracks[:, 0, 0], np.arange(45) + 1)
        np.testing.assert_allclose(tracks[:, 1, 0], np.arange(45) + 3)
        self.assertTrue(visibility.all())

    def test_terminal_key_reader_without_tty(self):
        read_descriptor, write_descriptor = os.pipe()
        with os.fdopen(read_descriptor) as stream, os.fdopen(write_descriptor, "w") as writer:
            with TerminalKeyReader(stream) as terminal:
                writer.write("N\n")
                writer.flush()
                self.assertEqual(terminal.poll(), "n")
                self.assertEqual(terminal.poll(), "enter")

    def test_selection_helpers(self):
        image = np.zeros((8, 8, 3), np.uint8)
        selected = flatten_direct_points(
            {"fake": image}, {"fake": [{"name": "hand", "points": [[2, 3], [4, 5]]}]},
            ["fake"],
        )
        np.testing.assert_array_equal(selected["fake"][0], [[2, 3], [4, 5]])
        mask = np.zeros((8, 8), bool)
        mask[1:7, 1:7] = True
        points, names = sample_mask_grid({"hand": mask}, mask.shape, grid_spacing=2)
        self.assertGreater(len(points), 0)
        self.assertEqual(set(names), {"hand"})

    def test_keypoint_overlay_is_saved_by_camera_and_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_keypoint_overlay(
                directory, "front/camera", 7,
                np.zeros((16, 16, 3), np.uint8),
                np.array([[4, 5]], np.float32), np.array([True]),
                np.array([0]), np.array(["hand"]),
            )
            self.assertTrue(path.is_file())
            self.assertEqual(path.name, "frame_000007.png")
            self.assertEqual(path.parent.name, "front_camera")

    def test_visualization_fps_and_video_export(self):
        timestamps = np.arange(24, dtype=np.float64) / 20.0
        np.testing.assert_allclose(visualization_fps(timestamps, [0, 8, 16]), 2.5)
        np.testing.assert_allclose(visualization_fps(timestamps, [0, 1, 2]), 20.0)
        alternating = np.cumsum([0.0] + [1 / 30, 2 / 30] * 8)
        np.testing.assert_allclose(
            selected_visualization_fps(
                alternating, np.arange(len(alternating)), 1, recording_fps=20.0
            ),
            20.0,
        )
        frames = np.zeros((3, 16, 16, 3), np.uint8)
        tracks = np.full((3, 1, 2), [4, 5], np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = write_keypoint_video(
                Path(directory) / "tracking.mp4", frames, np.arange(3),
                tracks, np.ones((3, 1), bool), np.array([0]), np.array(["hand"]),
                fps=20.0,
            )
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)

    def test_raw_and_tracked_hdf5(self):
        camera = FakeCamera()
        calibration = camera.initialize()
        raw = RGBDRecording("fake", calibration)
        tracked = TrackedRGBDRecording("fake", calibration)
        for index in range(2):
            frame = camera.read()
            raw.append(frame)
            result = TrackFrame(
                15 + 8 * index, np.array([[1, 1]], np.float32), np.array([True]),
                np.array([0]), np.array(["hand"]), "inferred",
            )
            tracked.append_tracking(
                frame, result,
                {"points": np.array([[0, 0, 2]], np.float32), "visible": np.array([True])},
            )
        with tempfile.TemporaryDirectory() as directory:
            raw_path = save_recording(Path(directory) / "raw.hdf5", {"fake": raw})
            tracked_path = save_recording(Path(directory) / "tracked.hdf5", {"fake": tracked})
            with h5py.File(raw_path, "r") as handle:
                self.assertEqual(handle["obs/fake/frames/rgb"].shape, (2, 4, 4, 3))
                self.assertEqual(handle["obs/fake/frames/depth"].shape, (2, 4, 4))
                np.testing.assert_array_equal(
                    handle["obs/fake/meta/intrinsics_heights_width"], [4, 4]
                )
            write_tracking_group(
                raw_path, "offline", "fake",
                np.zeros((1, 1, 2)), np.zeros((1, 1, 3)), np.ones((1, 1), bool),
                np.array([15]), np.array([0]), np.array(["hand"]),
                {"fake": [{"name": "hand", "points": [[1, 1]]}]},
            )
            with h5py.File(raw_path, "r") as handle:
                self.assertEqual(handle["tracking/offline/fake/keypoints_3d"].shape, (1, 1, 3))
            with h5py.File(tracked_path, "r") as handle:
                self.assertEqual(
                    handle["obs/fake/tracking/online/keypoints_3d"].shape, (2, 1, 3)
                )
                np.testing.assert_array_equal(
                    handle["obs/fake/tracking/online/tracker_frame_index"], [15, 23]
                )


if __name__ == "__main__":
    unittest.main()
