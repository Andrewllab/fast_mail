# Batched RGB-D online 3D keypoint tracking

This is the batched multi-view counterpart of `alex_3d`: one SAM3/CoTracker
pipeline instance handles all camera views, while each view is lifted with its
own RGB-D calibration. Use the same API and calibration format documented in
`../alex_3d/README.md`, importing classes from
`alex_batch_3d.track_online_pipe`.

```python
from alex_batch_3d.track_online_pipe import (
    CameraCalibration, SAM3Segmenter, OnlineKeypointTracker, OnlineTrackingPipeline,
)
pipeline = OnlineTrackingPipeline(
    SAM3Segmenter(cameras, device="cuda"),
    OnlineKeypointTracker(cameras, device="cuda"),
    {camera: CameraCalibration(K[camera], T[camera], depth_scale=0.001)
     for camera in cameras},
)
pipeline.initialize(first_rgb_images, selections)
for rgb_images, depth_images in stream:
    pipeline.push(rgb_images, depth_images)
pipeline.export("outputs/batched_3d")
```

Direct-click initialization is also supported. Configure replay with
`--set selection=keypoints --set 'objects=["hand","block"]'`, or call
`pipeline.initialize_points(first_rgb_images, selections)`. Selection entries
use `{"name": OBJECT_NAME, "points": [[x, y], ...]}` and may contain different
point counts across cameras.
