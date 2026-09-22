# D405 hand-keypoint capture and processing

The implementation has three executable variants and one shared camera API.

## Camera API

`BaseRGBDCamera.initialize()` starts a camera and returns calibration;
`read()` returns synchronized RGB, aligned optical-axis Z-depth in metres,
intrinsics, camera-to-world extrinsics, frame index and timestamps; `close()`
releases it. `RealSenseD405Camera` is the first implementation. Future cameras
should inherit the same base class.

Install the robot camera dependencies before using hardware:

```bash
conda activate ./.env
pip install -r requirements_robot.txt
```

Set the real D405 camera-to-world transform in the JSON configs. Identity is a
valid camera frame but is not a calibrated robot/world frame.

## Variant 1: live online tracking

Edit `alex_3d/config_d405_live.json`. Set `output_dir` to a directory to enable
HDF5 saving; leave it null to run tracking without storing anything. Selection
can be `keypoints`, `mask_click`, or `prompt`. `keypoints` skips SAM3 entirely.
`objects` may contain multiple unique names, for example `["hand", "cup"]`.
Direct-click mode opens one selection at a time in that order; terminal Enter
advances to the next object and begins tracking after the last object.

```bash
.env/bin/python alex_3d/live_track_d405.py \
  --config alex_3d/config_d405_live.json
```

Controls are typed in the launching terminal; the RGB window is display/mouse
only. On a normal terminal, these are immediate single-key commands and do not
require Enter:

- `n`: start a new episode and open the configured selection UI.
- `s`: save the current episode, report completion, and exit.
- `d`: discard the current episode and return to preview.
- `q`: quit.

For click-based mask or keypoint selection, click in the image window, return
focus to the terminal, and press Enter there to confirm. Press terminal `q` to
cancel selection. Saving prints progress and a completion message, then exits.

`record_frequency_hz` is the sparse output frequency. The pipeline samples the
camera at eight times that rate. The first stored sample appears after the
initial 16 tracker inputs; later stored samples are separated by eight inputs.
The HDF5 contains only these sparse RGB-D/keypoint samples.

## Variant 2: record first, online-track later

Edit `alex_3d/config_d405_record.json`. Again, null `output_dir` means preview
and timing-loop operation only: frames are not even buffered in memory and no
file is created.

```bash
.env/bin/python alex_3d/record_d405.py \
  --config alex_3d/config_d405_record.json
```

Use the same terminal `n/s/d/q` controls. A successful save prints the output
path and exits. This stores synchronized RGB-D frames,
intrinsics, per-frame camera-to-world transforms, indices and timestamps but no
keypoints. For a desired later keypoint rate of 2 Hz, record at 16 Hz because
online CoTracker emits one result for every eight new recorded frames.

Add online tracks to the same file:

```bash
.env/bin/python -m alex_3d.process_recording_online recordings/rgbd_....hdf5 \
  --selection keypoints --pipeline independent --visualize
```

Other initialization modes:

```bash
# SAM3 click mask -> masked grid
.env/bin/python -m alex_3d.process_recording_online FILE.hdf5 \
  --selection mask_click --objects hand

# SAM3 text mask -> masked grid
.env/bin/python -m alex_3d.process_recording_online FILE.hdf5 \
  --selection prompt --objects hand --prompt "human hand"
```

Results are written under `/tracking/online/<camera>`. Existing tracking data
is protected unless `--overwrite-tracking` is given.

The online predictor internally produces a dense finalized timeline at each
completed 16/8 update. Choose which covered frames are persisted with
`--first-output-frame` and `--output-step`. For example, store every covered
frame and save a matching RGB-overlay video:

```bash
.env/bin/python -m alex_3d.process_recording_online FILE.hdf5 \
  --selection keypoints --first-output-frame 0 --output-step 1 \
  --visualization-dir outputs/online_keypoints --overwrite-tracking
```

The final incomplete online tail is not predicted. With 20 recorded frames,
only frames through 15 are covered; with 24 frames, frames through 23 are
covered. Videos are written as `OUTPUT/online_<camera>.mp4`. Playback FPS uses
the nominal recording frequency stored in HDF5 divided by `output_step`, with
timestamps as fallback; `--visualization-fps` overrides it.

Add `--export-3d-dir OUTPUT --export-3d-fps RATE` to create the world-track
NPZ, final 3D plot, and animated 3D trajectory video. For a 20 Hz recording,
online outputs occur at 20/8 = 2.5 Hz, so use `--export-3d-fps 2.5`.

To use the repository's cached/local CoTracker code and an explicit checkpoint:

```bash
.env/bin/python -m alex_3d.process_recording_online FILE.hdf5 \
  --selection keypoints --hub-source local \
  --cotracker-repo .cache/torch/hub/facebookresearch_co-tracker_main \
  --checkpoint PATH/TO/model.pth
```

## Variant 3: full-video offline CoTracker3

This version gives CoTracker the complete saved video, so it can use future
frames. It tracks the first-frame queries across the whole sequence, then saves
frames 15, 23, 31, ... for cadence compatibility with the online pipeline.

For recordings that do not fit in GPU memory, set a bounded overlapping window:

```bash
.env/bin/python -m alex_3d.process_recording_offline FILE.hdf5 \
  --selection keypoints \
  --max-window-frames 64 --window-overlap-frames 16 \
  --first-output-frame 0 --output-step 1 \
  --overwrite-tracking
```

Each new window is initialized from the preceding window's prediction at the
shared boundary. Point ordering, IDs, and object names remain fixed across all
windows. For each point, the handoff prefers its most recent visible prediction
inside the overlap and passes that frame as the point's CoTracker query time;
an overlap-start coordinate is the fallback. The overlap therefore provides
both a stable identity anchor and temporal context. Only one RGB window is
transferred to the model at a time, and depth is read only for output frames.
`--first-output-frame 0 --output-step 1` stores every video frame; defaults `15`
and `8` retain cadence compatibility with the online pipeline.
The final overlap may be enlarged automatically so the last model call still
contains at least 16 frames.

Add `--visualization-dir OUTPUT` to save the rendered offline RGB-overlay video
as `OUTPUT/offline_<camera>.mp4`. This works with or without the interactive
`--visualize` window. Both saved and interactive visualization use the nominal
recording frequency divided by `output_step`; `--visualization-fps RATE`
overrides it.

```bash
.env/bin/python -m alex_3d.process_recording_offline FILE.hdf5 \
  --selection keypoints --visualize
```

Mask-grid prompt/click modes use the same flags as the online processor. For a
local CoTracker checkout:

```bash
.env/bin/python -m alex_3d.process_recording_offline FILE.hdf5 \
  --selection keypoints \
  --cotracker-repo .cache/torch/hub/facebookresearch_co-tracker_main \
  --hub-source local
```

Add `--checkpoint PATH/TO/model.pth` when you want an explicit local weight
file. Use `--max-frames 64` to cap the total video for a quick test; use
`--max-window-frames 64` to process the complete video in bounded chunks.

## HDF5 layout

Raw captures are under `/obs/<camera>/frames` with `rgb`, `depth`, `time`,
`hardware_time_ms`, `source_index`, and `global_extrinsics`. Intrinsics are at
`/obs/<camera>/meta/intrinsics`, with image size in
`intrinsics_heights_width`. This mirrors the nearby robot-data camera layout
while adding aligned metric `depth`. Live sparse tracking is saved beside
those frames under `/obs/<camera>/tracking/online`; post-processing results
are appended under `/tracking/online/<camera>` or
`/tracking/offline/<camera>`.

Offline results are stored under `/tracking/offline/<camera>`. Processing is
sequential across cameras to limit GPU memory, but a full camera video is held
for each model call. Use `--max-frames` for a short test.

## Hardware-free tests

```bash
.env/bin/python -m unittest alex_3d.test_streaming -v
.env/bin/python alex_3d/live_track_d405.py --help
.env/bin/python alex_3d/record_d405.py --help
.env/bin/python -m alex_3d.process_recording_online --help
.env/bin/python -m alex_3d.process_recording_offline --help
```
