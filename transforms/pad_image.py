import torch

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform
from utils.FoundationStereo.utils import InputPadder

class PadImage(Transform):
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, specs: DataSpecs) -> None:

        self._ir_keys = [key for key, spec in specs.obs.items() if spec.type == "ir"] # TODO: Handle IR stereo obs correctly: { "ir": { "left": img1, "right": img2 } } ?
        self._specs = specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=[("ir", key)],
                out_keys=[("ir", key)])
            for key in self._ir_keys
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, ir_obs):
        left = ir_obs["left"]
        right = ir_obs["right"]
        left = torch.as_tensor(left).cuda().float()[None].permute(0,3,1,2)
        img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
        padder = InputPadder(left.shape, divis_by=32, force_square=False)
        left, right = padder.pad(left, right)
        return left, right # TODO: Handle left and right correclty! Dict?