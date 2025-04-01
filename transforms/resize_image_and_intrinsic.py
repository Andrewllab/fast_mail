import torch
import cv2

from environments.specs import DataSpecs
from transforms.base_transform import KeyMapping, Transform


class ResizeImageAndIntrinsic(Transform):
    def __init__(self, specs: DataSpecs, target_height, target_width) -> None:

        self.target_height = target_height # e.g. 480
        self.target_width = target_width # e.g. 640
        self._ir_keys = [key for key, spec in specs.obs.items() if spec.type == "ir"] # TODO: Handle IR stereo obs correctly: { "ir": { "left": img1, "right": img2 } } ?
        self._specs = specs
        # TODO: add intrinsic via CameraSpec!

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
    
    # @property # TODO: How to handle intrinsic updates correctly?
    # def intrinsic_specs(self) -> CameraSpec:
    #     return self._specs

    def __call__(self, ir_obs):
        #for side in image:
        ir_obs['left'], intrinsic = resize_cover_and_crop_center(ir_obs['left'], self.intrinsic, self.target_height, self.target_width)
        ir_obs['right'], _ = resize_cover_and_crop_center(ir_obs['right'], self.intrinsic, self.target_height, self.target_width)

        return ir_obs, intrinsic # TODO: how to handle intrinsic updates here correctly?


def resize_cover_and_crop_center(img, K, target_h=480, target_w=640):
    h, w = img.shape[:2]

    # Resize to make *both* dimensions ≥ target
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize without stretching
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Center crop to target size
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    cropped_img = resized_img[start_y:start_y + target_h, start_x:start_x + target_w]

    # Update intrinsics matrix K for resized+cropped image
    new_K = K.copy()
    new_K[0:2] *= scale # Update fx, fy (scale focal length)

    # Update cx, cy (move principal point due to crop)
    dx = (new_w - target_w) // 2
    dy = (new_h - target_h) // 2
    new_K[0, 2] -= dx  # cx -= crop on x
    new_K[1, 2] -= dy  # cy -= crop on y

    return cropped_img, new_K