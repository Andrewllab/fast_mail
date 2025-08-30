"""Create ONNX model out of FoundationStereo for compiling into TensorRT engine

Modified from third_party/FoundationStereo/scripts/make_onnx.py
"""

import argparse
import logging
import os

import rootutils
import torch
from omegaconf import OmegaConf

# enables importing local modules regardless of where the script is run
rootutils.setup_root(__file__, indicator=".isort.cfg", pythonpath=True)

from third_party.FoundationStereo.core.foundation_stereo import FoundationStereo
from third_party.FoundationStereo.core.utils.utils import InputPadder

log = logging.getLogger("make_onnx")


class FoundationStereoOnnx(FoundationStereo):
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def forward(self, left, right):
        """Removes extra outputs and hyper-parameters"""
        with torch.amp.autocast("cuda", enabled=True):
            disp = FoundationStereo.forward(
                self, left, right, iters=self.args.valid_iters, test_mode=True
            )
        return disp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, help="Path to save results.")
    parser.add_argument("--ckpt_dir", type=str, help="pretrained model path")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=16,
        help="number of flow-field updates during forward pass",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="number of stereo pairs to process at once",
    )
    args = parser.parse_args()

    # add convenient short cuts for "large" and "small"
    ckpt_dir = args.ckpt_dir
    if ckpt_dir == "large":
        ckpt_dir = "foundation_stereo_models/23-51-11/model_best_bp2.pth"
    elif ckpt_dir == "small":
        ckpt_dir = "foundation_stereo_models/11-33-40/model_best_bp2.pth"

    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    torch.autograd.set_grad_enabled(False)

    cfg = OmegaConf.load(f"{os.path.dirname(ckpt_dir)}/cfg.yaml")
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    if "vit_size" not in cfg:
        cfg["vit_size"] = "vitl"
    args = OmegaConf.create(cfg)
    log.info(f"args:\n{args}")
    log.info(f"Using pretrained model from {ckpt_dir}")
    model = FoundationStereoOnnx(cfg)
    ckpt = torch.load(ckpt_dir, weights_only=False)
    log.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt["model"])
    model.cuda()
    model.eval()

    batch_size = args.batch_size
    left_img = torch.randn(batch_size, 3, args.height, args.width).cuda().float()
    right_img = torch.randn(batch_size, 3, args.height, args.width).cuda().float()

    # pad the image to ensure it's a shape that is compatible with FoundationStereo
    padder = InputPadder((args.height, args.width), divis_by=32, force_square=False)
    left_img, right_img = padder.pad(left_img, right_img)

    torch.onnx.export(
        model,
        (left_img, right_img),
        args.save_path,
        opset_version=16,
        input_names=["left", "right"],
        output_names=["disp"],
        # # If these are enabled, trtexec must be given --optShapes argument to
        # # specify the input shapes including batch size, otherwise it assumes
        # # a batch size of 1
        # dynamic_axes={
        #     "left": {0: "batch_size"},
        #     "right": {0: "batch_size"},
        #     "disp": {0: "batch_size"},
        # },
    )

    log.info(f"ONNX model exported at {args.save_path}")
    log.warning(f"Final image dimensions of exported model are {left_img.shape[-2:]}")
