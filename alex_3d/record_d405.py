"""Interactive D405 RGB-D recorder without keypoint tracking.

Terminal controls: ``n`` starts, ``s`` saves, ``d`` discards, and ``q`` exits.
If ``output_dir`` is absent/null, the loop runs and previews but never stores.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alex_3d.cameras import camera_from_config
from alex_3d.hdf5_io import RGBDRecording, next_episode_path, save_recording
from alex_3d.live_utils import FrequencyGate, load_json, overlay_status
from alex_3d.terminal_input import TerminalKeyReader


def run(config):
    camera = camera_from_config(config["camera"])
    calibration = camera.initialize()
    frequency = float(config.get("capture_frequency_hz", 16.0))
    if frequency > camera.fps + 1e-6:
        raise ValueError(f"capture_frequency_hz={frequency} exceeds camera fps={camera.fps}")
    gate = FrequencyGate(frequency)
    output_dir = config.get("output_dir")
    output_dir = None if output_dir in (None, "") else Path(output_dir)
    recording = None
    recording_active = False
    captured_frames = 0
    window = f"RGB-D recorder: {camera.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Terminal controls: n=new recording, s=save, d=discard, q=quit.", flush=True)
    try:
        with TerminalKeyReader() as terminal:
            while True:
                frame = camera.read()
                if recording_active and gate.due():
                    captured_frames += 1
                    if recording is not None:
                        recording.append(frame)
                text = (f"RECORDING frames={captured_frames} | terminal: s save, d discard"
                        if recording_active else "PREVIEW | terminal: n start, q quit")
                cv2.imshow(window, overlay_status(frame.rgb, text))
                cv2.waitKey(1)
                command = terminal.poll()
                if command == "q":
                    print("Quit requested. Exiting without saving the active recording.", flush=True)
                    break
                if not recording_active and command == "n":
                    recording = (
                        RGBDRecording(camera.name, calibration) if output_dir is not None else None
                    )
                    recording_active = True
                    captured_frames = 0
                    gate.reset()
                    print(f"RECORDING STARTED at {frequency:g} Hz. Press s to save or d to discard.", flush=True)
                elif recording_active and command == "d":
                    print(f"Recording discarded ({captured_frames} frames). Ready for n or q.", flush=True)
                    recording = None
                    recording_active = False
                    captured_frames = 0
                elif recording_active and command == "s":
                    if output_dir is None:
                        print("Cannot save: output_dir is not configured.", flush=True)
                    elif recording is None or not len(recording):
                        print("Cannot save: no frames have been captured yet.", flush=True)
                    else:
                        path = next_episode_path(output_dir, "rgbd")
                        print(f"SAVING {len(recording)} RGB-D frames to {path} ...", flush=True)
                        save_recording(path, {camera.name: recording}, metadata={
                            "capture_frequency_hz": frequency,
                        })
                        print(f"SAVE COMPLETE: {path}. Exiting.", flush=True)
                        break
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
