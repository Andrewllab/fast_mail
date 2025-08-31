import argparse
import time

import rootutils
import torch

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from third_party.FoundationStereo.core.utils.utils import InputPadder
from utils.tensor_rt import load_engine, run_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--engine_path", type=str, help="Path to the TensorRT engine file."
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="number of stereo pairs to process at once",
    )
    args = parser.parse_args()

    engine, context = load_engine(args.engine_path, log_level="VERBOSE")

    # load example png images from filepath and convert to tensor
    left = torch.rand(args.batch_size, 3, args.height, args.width, device="cuda")
    right = torch.rand(args.batch_size, 3, args.height, args.width, device="cuda")

    # scale to [0, 255]
    left.mul_(255.0)
    right.mul_(255.0)

    padder = InputPadder(left.shape[-2:], divis_by=32, force_square=False)
    left, right = padder.pad(left, right)

    durations = []
    for _ in range(10):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        disp = run_inference(engine, context, left=left, right=right)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        duration = end_time - start_time
        print(f"Inference time: {duration:.3f} seconds")
        durations.append(duration)

    print(f"Average inference time: {sum(durations) / len(durations):.3f} seconds")
