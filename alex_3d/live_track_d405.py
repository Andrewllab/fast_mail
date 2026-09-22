"""Interactive D405 RGB-D hand/object tracking and sparse HDF5 recording.

Terminal controls: ``n`` starts a new selection/tracking episode and ``q`` exits.
During an episode, ``s`` saves and ``d`` discards.  Stored samples begin at the
first completed CoTracker window (input frame 15) and continue every eight
tracker inputs.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alex_3d.cameras import camera_from_config
from alex_3d.hdf5_io import TrackedRGBDRecording, next_episode_path, save_recording
from alex_3d.live_utils import FrequencyGate, load_json, overlay_status, selections_for_mode
from alex_3d.selection import DirectPointSegmenter
from alex_3d.terminal_input import TerminalKeyReader


def _device(value):
    return ("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value


def _pipeline(config, camera, calibration, first, selections):
    module_name = "alex_batch_3d.track_online_pipe" if config.get("pipeline") == "batch" else "alex_3d.track_online_pipe"
    api = __import__(module_name, fromlist=["SAM3Segmenter", "OnlineKeypointTracker", "OnlineTrackingPipeline"])
    name = camera.name
    device = _device(config.get("device", "auto"))
    tracker_options = dict(config.get("tracker", {}))
    local_repo = tracker_options.pop("local_repo", None)
    checkpoint = tracker_options.pop("checkpoint", None)
    if local_repo:
        sys.path.insert(0, str(Path(local_repo).resolve()))
        from cotracker.predictor import CoTrackerOnlinePredictor
        tracker_options["predictor_factory"] = lambda: CoTrackerOnlinePredictor(checkpoint=checkpoint)
    tracker = api.OnlineKeypointTracker([name], device=device, **tracker_options)
    mode = config.get("selection", "keypoints")
    if mode == "keypoints":
        segmenter = DirectPointSegmenter([name])
    else:
        segmenter = api.SAM3Segmenter(
            [name], device=device, **config.get("segmenter", {})
        )
    pipeline = api.OnlineTrackingPipeline(
        segmenter, tracker, {name: calibration}, keep_history=False
    )
    images = {name: first.rgb}
    if mode == "keypoints":
        pipeline.initialize_points(images, selections)
    else:
        pipeline.initialize(images, selections)
    return api, pipeline, tracker


def run(config):
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    camera = camera_from_config(config["camera"])
    output_dir = config.get("output_dir")
    output_dir = None if output_dir in (None, "") else Path(output_dir)
    final_hz = float(config.get("record_frequency_hz", 2.0))
    tracker_input_hz = final_hz * 8.0
    if tracker_input_hz > camera.fps + 1e-6:
        raise ValueError(
            f"record_frequency_hz={final_hz} needs {tracker_input_hz} tracker inputs/s, "
            f"but the camera is configured for {camera.fps} fps"
        )
    visualize = bool(config.get("visualize_keypoints", True))
    objects = list(config.get("objects", ["hand"]))
    prompts = dict(config.get("prompts", {"hand": "human hand"}))
    window = f"RGB-D tracking: {camera.name}"
    state = "preview"
    api = pipeline = tracker = recording = selections = None
    gate = FrequencyGate(tracker_input_hz)

    calibration = camera.initialize()
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Terminal controls: n=new episode, s=save, d=discard, q=quit.", flush=True)
    try:
        with TerminalKeyReader() as terminal:
            while True:
                frame = camera.read()
                display = overlay_status(frame.rgb, "PREVIEW: terminal n start | q quit")
                if state == "tracking":
                    if gate.due():
                        pipeline.push({camera.name: frame.rgb}, {camera.name: frame.depth})
                        if tracker.updated:
                            latest_2d = pipeline.get_latest_keypoints(camera.name)
                            latest_3d = pipeline.get_latest_3d_keypoints(camera.name)
                            if recording is not None:
                                recording.append_tracking(frame, latest_2d, latest_3d)
                            print(
                                f"tracked frame={latest_2d.frame_index} "
                                f"visible={int(latest_3d['visible'].sum())}/{len(latest_3d['visible'])}",
                                flush=True,
                            )
                            if visualize:
                                display = cv2.cvtColor(
                                    api.OnlineKeypointTracker.visualize(
                                        {camera.name: frame.rgb},
                                        {camera.name: latest_2d},
                                        show_ids=True,
                                    )[camera.name],
                                    cv2.COLOR_RGB2BGR,
                                )
                    count = 0 if recording is None else len(recording)
                    cv2.putText(
                        display,
                        f"TRACKING samples={count} | terminal: s save, d discard",
                        (6, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 220, 255), 1, cv2.LINE_AA,
                    )
                cv2.imshow(window, display)
                cv2.waitKey(1)
                command = terminal.poll()
                if command == "q":
                    print("Quit requested. Exiting without saving the active episode.", flush=True)
                    break
                if state == "preview" and command == "n":
                    print("NEW EPISODE: select initialization points/mask in the image.", flush=True)
                    selection_mode = config.get("selection", "keypoints")
                    selections = selections_for_mode(
                        selection_mode, {camera.name: frame.rgb}, objects, prompts,
                        terminal=terminal,
                    )
                    api, pipeline, tracker = _pipeline(
                        config, camera, calibration, frame, selections
                    )
                    recording = None
                    if output_dir is not None:
                        recording = TrackedRGBDRecording(camera.name, calibration)
                        recording.selection = selections
                    gate.reset()
                    state = "tracking"
                    print(
                        "TRACKING STARTED: collecting 15 more inputs for warmup; "
                        "s=save, d=discard.",
                        flush=True,
                    )
                elif state == "tracking" and command == "s":
                    if output_dir is not None and recording is not None and len(recording):
                        path = next_episode_path(output_dir, "tracked")
                        print(
                            f"SAVING {len(recording)} tracked RGB-D samples to {path} ...",
                            flush=True,
                        )
                        save_recording(path, {camera.name: recording}, metadata={
                            "selection": selections,
                            "record_frequency_hz": final_hz,
                            "tracker_input_frequency_hz": tracker_input_hz,
                            "pipeline": config.get("pipeline", "independent"),
                        })
                        print(f"SAVE COMPLETE: {path}. Exiting.", flush=True)
                        break
                    if output_dir is None:
                        print("Cannot save: output_dir is not configured.", flush=True)
                    else:
                        print("Cannot save until CoTracker has produced its first result.", flush=True)
                elif state == "tracking" and command == "d":
                    print("Episode discarded. Ready for n or q.", flush=True)
                    state = "preview"
                    api = pipeline = tracker = recording = selections = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    finally:
        cv2.destroyAllWindows()
        camera.close()


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(arguments)
    run(load_json(args.config))


if __name__ == "__main__":
    main()
