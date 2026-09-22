# Multi-camera online 2D point tracking

SAM3 selects objects on the first frame; CoTracker3 online tracks fixed masked-grid
queries across subsequent frames. All coordinates are `(x, y)` pixels in the
processed RGB images. No Hydra, depth, robot actions, or frequency simulation.

## Streaming behavior

`OnlineKeypointTracker` owns the buffering and startup logic for every camera.
It uses the standard **16-frame window and eight-frame stride**:

| Received frames (zero-based) | Action | Latest result |
| --- | --- | --- |
| 0 | Select masks/grid, initialize model queries, buffer first image | None |
| 1–14 | Buffer only | None |
| 15 | Infer frames 0–15; retain frames 8–15 | Frame 15 |
| 16–22 | Buffer eight new images | Frame 15 |
| 23 | Infer frames 8–23; retain frames 16–23 | Frame 23 |
| 24–30 | Buffer | Frame 23 |
| 31 | Infer frames 16–31 | Frame 31 |

The first inference prints:
`Starting CoTracker3 online inference: 16 frames accumulated.`

There are no partial-window previews or state snapshots. The native online
predictor retains its state and query IDs across full-window calls. Each camera
has its own predictor. Inference is sequential across views.

`get_latest_keypoints()` returns only the last inferred frame for each camera,
not the full track history. It returns `None` before the initial 16 frames.
Between inference calls it returns the previous result, whose `frame_index`
identifies the image it belongs to. `get_latest_keypoints("left_cam")` returns a
single camera's result. Returned arrays are copies, so callers cannot corrupt
the stored state.

`push(images)` adds one frame from each camera and runs inference only when due;
it also returns the latest result for convenience. `tracker.updated` says whether
this push ran inference. `tracker.frame_index` counts received frames;
`tracker.inference_count` counts updates per camera.

`finish()` does not infer an incomplete tail. For 26 input frames, inference
covers frames 0–23 and the last two remain unprocessed. For a stream shorter than
16 frames, no tracks are available. For the full 454-frame example, 55 inference
calls per camera cover 448 frames; the final six frames remain unprocessed.

## Classes

- `SAM3Segmenter` in `track_online_pipe.py`: first-frame text/click masks,
  interactive click collection, mask overlays, and optional model release.
- `OnlineKeypointTracker` in the same file: query sampling, 16/8 buffering,
  persistent multi-camera tracking, latest-keypoint getter, and overlays.
- `OnlineTrackingPipeline` in the same file: segmentation/tracking coordination,
  timing, recording, current-result visualization, NPZ and video export.
- `HDF5FrameReader` in `simulate_static.py`: reads only RGB image datasets at
  its current pointer and advances it when `next_frame()` is called.

The reader has no prefetch queue, timestamp reads, playback clock, or sleep.
`next_frame()` returns a `FrameBundle` with `images`, sequential `index`,
recording `source_index`, and `original_sizes_wh`; EOF returns `None`.
Images are contiguous RGB uint8 HWC. Optional resizing, BGR conversion,
grayscale/RGBA conversion and explicit float-range conversion are supported.
`source.pointer` is the index of the next selected bundle.

The example recording contains 454 frames in three 256x256 views. Automatic
discovery uses `obs/gripper_cam/frames/rgb` and
`obs/{left_cam,right_cam}/frames/left`. Images are paired by row index.

## Run

From the repository root, inspect RGB inputs without loading any model:

```bash
.env/bin/python alex/simulate_static.py --inspect
```

Direct-click tracking bypasses SAM3. The object list controls the selection
order; after clicking points for one object, press Enter in the launching
terminal to advance to the next object (and then the next camera):

```bash
.env/bin/python alex/simulate_static.py \
  --set selection=keypoints \
  --set 'objects=["hand","block"]'
```

Programmatically, use the same named selection structure with
`pipeline.initialize_points(first_rgb_images, selections)`. Each selection is
`{"name": "hand", "points": [[x, y], ...]}`. This API does not load SAM3.

Text selection of red and blue cups in both external views:

```bash
.env/bin/python alex/simulate_static.py \
  --config alex/example_text.json \
  --set tracker.grid_spacing=16 \
  --set source.stop=40 \
  --set output_dir=alex/outputs/cups
```

RGB-D 3D replay (the example cup file has no metric depth, so this uses a
clearly non-metric synthetic-depth smoke test):

```bash
.env/bin/python alex/simulate_static_3d.py \
  --config alex/example_text.json \
  --synthetic-depth \
  --output-dir alex_3d/outputs/cups_replay
```

For real metric output, replace `--synthetic-depth` with one
`--depth-path CAMERA=HDF5_DATASET` per camera and provide `--calibration-json`.

Remove `source.stop=40` to read the whole recording. Omit `--config` for interactive
click selection. Left click adds foreground, right click adds background, and
Enter accepts each camera/object selection. A GUI display is needed only for
click selection and explicit display options.

The replay writes `*_sam3_cotracker.mp4` and matching `*_sam3_cotracker.npz`
files. These contain only completed CoTracker windows (frames 15, 23, 31, ...),
so buffered frames are not duplicated with stale keypoints. Text selections use
`{"name":"cup","text":"red cup"}`; click selections use
`{"name":"cup","points":[[x,y]],"labels":[1]}` per camera.

To compare SAM3 prompt and click masks on the initial frame in one PNG:

```bash
.env/bin/python alex/test_segmenter.py \
  --prompt 'red cup' \
  --clicks-json '{"left_cam":{"cup":{"points":[[165,85]],"labels":[1]}},"right_cam":{"cup":{"points":[[92,108]],"labels":[1]}}}' \
  --output alex/outputs/segmenter_test.png \
  --masks-output alex/outputs/segmenter_test_masks.npz
```

The saved image has prompt overlays in the first row, click overlays in the
second row, and one column per camera. If `--clicks-json` is omitted, the script
uses the center pixel of each image as a simple foreground-click fallback; for a
meaningful test, provide clicks on the selected object. The prompt and click
models are loaded lazily and cached within the test process.

Configuration precedence: `DEFAULT_CONFIG` < optional JSON file < repeated
`--set dotted.key=JSON_VALUE`. Strings can be unquoted scalars. Examples:

```bash
--set 'objects=["cup","bowl"]'
--set selection=keypoints
--set 'source.camera_paths={"left_cam":"obs/left_cam/frames/left"}'
--set source.resize_wh='[256,256]'
--set tracker.max_points_per_object=64
--set device=cuda
--set show_initial=true
--set show_live=true
--set play_final=true
--set save_videos=false
```

`fps` only sets exported video playback speed; it never paces input reading.
The source's optional `start`, `stop` and `stride` select recording rows.
Segmentation always uses the first selected row.

Per-camera selection dictionaries support:

```json
{
  "left_cam": [
    {"name": "cup", "text": "red cup"},
    {"name": "bowl", "points": [[120, 90], [140, 130]], "labels": [1, 0]}
  ]
}
```

Use either text or clicks for each object. Text defaults to the highest-scoring
instance; `"instances":"all"` keeps separate masks named `name:0`, `name:1`, etc.
Clicks refer to the processed image dimensions. Provide a selection list for
every selected camera, and exclude cameras where no desired object is visible.
The resolved config, including interactive clicks, is saved for reuse.

## Live-camera interface

```python
from alex.track_online_pipe import (
    SAM3Segmenter, OnlineKeypointTracker, OnlineTrackingPipeline,
)

cameras = ["left", "right"]
pipeline = OnlineTrackingPipeline(
    SAM3Segmenter(cameras, device="cuda"),
    OnlineKeypointTracker(cameras, device="cuda"),
    keep_history=True,
)
pipeline.initialize(first_rgb_images, selections)

for rgb_images in camera_stream:
    pipeline.push(rgb_images)
    latest = pipeline.get_latest_keypoints()
    if latest is not None:
        left = latest["left"]
        pixels = left.xy[left.visible]
        ids = left.point_ids[left.visible]
        inferred_frame_index = left.frame_index

pipeline.finish()
pipeline.export("alex/outputs/live", save_videos=True)
```

The loop begins with the frame AFTER initialization. Each input is a
`{camera_name: RGB_array}` mapping containing all configured cameras. Camera
resolutions may differ, but must remain fixed individually. Capture alignment
is the caller's responsibility.

Each result has `frame_index`, `xy [N,2]`, `visible [N]`, `point_ids [N]`,
`object_names [N]`, and `status="inferred"`. IDs and ordering remain fixed per
camera even during occlusion. Use visibility to filter unreliable positions.
Overlapping masks may create duplicate positions associated with distinct objects.
No cross-camera correspondence, object re-detection or ID insertion is performed.

## Timing, storage and visualization

The loop prints a ticker for every received frame and writes `timings.csv` with
received/source indices, last inferred index, `inference_ran`, and
`inference_ms`. `inference_ms` measures only the wall-clock duration of the
CoTracker3 predictor call for that update; it excludes segmentation, frame
reading, visualization, and pipeline overhead. No CUDA synchronization is used.

CoTracker's complete predicted timeline is stored in `tracker.tracks[camera]`
and `tracker.visibility[camera]`, replacing previous predictions when overlap
is refined. The getter still exposes only its final frame.

With `keep_history=True`, the pipeline records RGB and exports:
- Per-view `*_tracks.npz`: `tracks [T_inferred,N,2]`, `visibility [T_inferred,N]`,
  fixed IDs/object names, frame indices, statuses, timing and image size.
- Per-view `*_tracks.mp4`: matched RGB frames, keypoints and recent trails.
- The demo also saves first RGBs, mask overlays/NPZs, final keypoint overlays,
  resolved configuration, source frame metadata and artifact paths.

Exports use the latest inferred timeline, including overlap refinements.
They exclude unprocessed tails; stale results are never assigned to newer images.
`visualize_current()` shows the last inferred image with its matching keypoints.
During warmup it shows the received RGB without keypoints. Buffered frames have
`inference_ms=null` because no CoTracker call occurs.

Set `keep_history=False` for live use without RGB/CPU export history. The tracker
still retains CoTracker's growing prediction timeline. Point identity bookkeeping
is persistent; model drift or difficult occlusions can still cause tracking errors.

## Dependencies and validation

Use CUDA-compatible PyTorch/torchvision and `alex/requirements.txt`. SAM requires
Hugging Face access approval for `facebook/sam3` and `hf auth login`. The demo
defaults model caches to `alex/.cache/`. Use `tracker.hub_source=local` and
`tracker.hub_repo=/path/to/co-tracker` for an existing checkout.

```bash
.env/bin/python -m unittest alex.test_online -v
.env/bin/python alex/test_cotracker.py
```

The diagnostic uses real CoTracker online weights and manual red-cup ROI masks
in both external views. It is explicitly NOT a SAM segmentation test. Its output
directory is `alex/outputs/cotracker_episode/`. It reads the complete episode and
writes only the inferred frames (15, 23, 31, …), so buffered frames are absent
from the video rather than represented by repeated keypoints.

SAM3 learned inference was previously blocked by Hugging Face HTTP 403 for the
available account. Unit tests also exercise the actual click processor using
simulated logits, without requiring model downloads.

References:
- https://github.com/facebookresearch/co-tracker/blob/main/online_demo.py
- https://github.com/facebookresearch/co-tracker/blob/main/cotracker/predictor.py
- https://huggingface.co/docs/transformers/model_doc/sam3
- https://huggingface.co/docs/transformers/model_doc/sam3_tracker
