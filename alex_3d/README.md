# RGB-D online 3D keypoint tracking

This extends `alex`'s independent-per-camera CoTracker3 online pipeline. SAM3
selects first-frame masks, grid points are sampled inside each mask, and the
last prediction from every completed 16-frame window is lifted into world
coordinates. The output cadence is therefore input frames 15, 23, 31, ...;
the RGB 2D exports remain unchanged.

The pipeline requires one depth image and calibration per camera. Calibration
can be a `CameraCalibration` or a mapping with `intrinsics` (3x3), `extrinsics`
(camera-to-world 4x4), optional `depth_scale` (default 1), and optional
`orthographic` (default false).

```python
from alex_3d.track_online_pipe import (
    CameraCalibration, SAM3Segmenter, OnlineKeypointTracker, OnlineTrackingPipeline,
)

calibrations = {
    "left_cam": CameraCalibration(K_left, T_left_camera_to_world, depth_scale=0.001),
    "right_cam": CameraCalibration(K_right, T_right_camera_to_world, depth_scale=0.001),
}
pipeline = OnlineTrackingPipeline(
    SAM3Segmenter(["left_cam", "right_cam"], device="cuda"),
    OnlineKeypointTracker(["left_cam", "right_cam"], device="cuda"),
    calibrations,
)
pipeline.initialize(first_rgb_images, selections)
for rgb_images, depth_images in stream:
    pipeline.push(rgb_images, depth_images)
pipeline.export("outputs/hand_3d")
```

To bypass SAM3 and track exact clicked pixels, use a segmenter adapter and the
native direct-point initializer:

```python
from alex_3d.selection import DirectPointSegmenter

pipeline = OnlineTrackingPipeline(
    DirectPointSegmenter(["left_cam", "right_cam"]), tracker, calibrations
)
pipeline.initialize_points(first_rgb_images, {
    "left_cam": [
        {"name": "hand", "points": [[120, 80], [135, 92]]},
        {"name": "block", "points": [[170, 140]]},
    ],
    "right_cam": [
        {"name": "hand", "points": [[95, 84]]},
        {"name": "block", "points": [[145, 138]]},
    ],
})
```

The replay CLI accepts `--set selection=keypoints --set
'objects=["hand","block"]'`. Interactive Enter confirmation is read from the
launching terminal and advances object-by-object, then camera-by-camera.

`depth_images` is a mapping of camera name to a `[H,W]` depth image aligned to
that camera's RGB image. `get_latest_3d_keypoints()` returns only the most
recent completed-window point set. `export_3d()` writes one world-track NPZ,
one final 3D snapshot, and a keypoint-only 3D MP4 per camera. The video frames
are the completed-window outputs, so consecutive frames normally represent an
8-input-frame interval.
