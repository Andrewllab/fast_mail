# Batched camera tracking

This folder is a parallel implementation of `alex` and does not modify the
original code. `SAM3Segmenter` batches all text jobs into one SAM3 call and all
click jobs into one SAM3 tracker call. `OnlineKeypointTracker` uses one
CoTracker3 online predictor with camera views as the batch dimension.

Run text-prompt replay:

```bash
python -m alex_batch.simulate_static --config alex_batch/example_text.json \
  --set output_dir=alex_batch/outputs/cups
```

Omit `selections` to use the inherited interactive click selector. Click lists
may have different lengths across cameras; they are padded with ignored labels.
Camera images may have different spatial sizes; the tracker pads them to the
largest batch size and clips returned visibility to each camera's original
size. Per-camera object grids may contain different numbers of points; query
slots are padded and removed from each camera's result.

The replay writes sparse `*_sam3_cotracker.mp4` videos and matching `.npz`
tracks only for completed online windows (15, 23, 31, ...).

To compare batched prompt and click masks on the first frame:

```bash
python -m alex_batch.test_segmenter --prompt "red cup" \
  --clicks-json '{"left_cam":{"object":{"points":[[160,90]],"labels":[1]}},"right_cam":{"object":{"points":[[100,110]],"labels":[1]}}}'
```

The diagnostic CoTracker replay (without SAM3 weights) is:

```bash
python -m alex_batch.test_cotracker
```

For RGB-D 3D replay with the batched CoTracker pipeline:

```bash
.env/bin/python alex_batch/simulate_static_3d.py \
  --config alex_batch/example_text.json \
  --synthetic-depth \
  --output-dir alex_batch_3d/outputs/cups_replay
```

Use `--depth-path CAMERA=HDF5_DATASET` and `--calibration-json` for metric
RGB-D data instead of `--synthetic-depth`.
