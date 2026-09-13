"""Smoke tests for batched SAM3/CoTracker3 behavior."""

import unittest

import numpy as np
import torch

if __package__:
    from .track_online_pipe import OnlineKeypointTracker
else:
    from track_online_pipe import OnlineKeypointTracker


class _Model:
    window_len = 16


class FakeBatchPredictor:
    step = 8
    v2 = False
    model = _Model()
    instances = []

    def __init__(self):
        self.calls = []
        self.queries = None
        self.timeline = 0
        FakeBatchPredictor.instances.append(self)

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, video, is_first_step=False, queries=None, add_support_grid=True):
        self.calls.append((tuple(video.shape), is_first_step))
        if is_first_step:
            self.queries = queries.detach().cpu()
            self.timeline = 1
            return None, None
        self.timeline = max(self.timeline + 8, int(video.shape[1]))
        batch, _, points, _ = self.queries.shape[0], self.timeline, self.queries.shape[1], 2
        xy0 = self.queries[:, :, 1:].detach().cpu().numpy()
        tracks = np.repeat(xy0[:, None], self.timeline, axis=1)
        tracks += np.arange(self.timeline, dtype=np.float32)[None, :, None, None]
        visible = np.ones((batch, self.timeline, points), dtype=bool)
        return torch.from_numpy(tracks), torch.from_numpy(visible)


class BatchedTrackerTests(unittest.TestCase):
    def setUp(self):
        FakeBatchPredictor.instances.clear()

    def test_single_predictor_and_unequal_point_counts(self):
        cameras = ("left", "right")
        images = {"left": np.zeros((32, 40, 3), np.uint8), "right": np.zeros((24, 48, 3), np.uint8)}
        masks = {"left": {"cup": np.ones((32, 40), bool)},
                 "right": {"cup": np.ones((24, 48), bool), "bowl": np.ones((24, 48), bool)}}
        tracker = OnlineKeypointTracker(cameras, device="cpu", grid_spacing=8,
                                        max_points_per_object=4, predictor_factory=FakeBatchPredictor)
        tracker.initialize(images, masks)
        self.assertEqual(len(FakeBatchPredictor.instances), 1)
        self.assertIs(tracker.predictors["left"], tracker.predictors["right"])
        for index in range(1, 15):
            self.assertIsNone(tracker.push(images))
            self.assertFalse(tracker.updated)
        result = tracker.push(images)
        self.assertTrue(tracker.updated)
        self.assertEqual(result["left"].xy.shape[0], 4)
        self.assertEqual(result["right"].xy.shape[0], 8)
        self.assertEqual(result["left"].frame_index, 15)
        self.assertEqual(len(FakeBatchPredictor.instances[0].calls), 2)
        self.assertEqual(FakeBatchPredictor.instances[0].calls[-1][0][0], 2)


if __name__ == "__main__":
    unittest.main()
