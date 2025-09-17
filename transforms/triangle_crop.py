from __future__ import annotations
from typing import Any

from dataclasses import dataclass
import torch
from tensordict import TensorDict

from environments.specs import CameraSpec, PointMapStream, DataSpecs
from transforms.base_transform import Transform

@dataclass
class TriangleCropBorder:
    point1: tuple[int, int]
    point2: tuple[int, int]
    mask_right: bool
    
    def in_img(self, height: int, width: int) -> bool:
        y1, x1 = self.point1
        y2, x2 = self.point2
        return 0 <= y1 < height and 0 <= x1 < width and 0 <= y2 < height and 0 <= x2 < width
    
    def build_mask(self, height: int, width: int) -> torch.Tensor:
        # Create coordinate grids
        y_coords = torch.arange(height).view(-1, 1).expand(-1, width)
        x_coords = torch.arange(width).view(1, -1).expand(height, -1)

        # Extract coordinates from points (y, x) format
        y1, x1 = self.point1
        y2, x2 = self.point2

        # Calculate the line equation: (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1) = 0
        # This is a cross product to determine which side of the line a point is on
        line_values = (y_coords - y1) * (x2 - x1) - (x_coords - x1) * (y2 - y1)

        # Create the mask based on the mask_right flag
        if self.mask_right:
            # Mask out points where line_values < 0 (right of the line)
            mask = line_values < 0
        else:
            # Mask out points where line_values > 0 (left of the line)
            mask = line_values >= 0

        return mask
    
class TriangleCrop(Transform):
    """
    Transform that masks out pixels on one side of a line defined by two points.

    If mask_right is True, all pixels right of the line are masked out.
    If mask_right is False, all pixels left of the line are masked out.

    The line is defined by two points, each with (y, x) coordinates.
    """

    def __init__(
        self,
        specs: DataSpecs,
        borders: dict[str, Any],
        mask_value: float = 0.0,
    ) -> None:
        # Store the input parameters
        self._output_specs = specs
        self.mask_value = mask_value

        # The output specs are the same as the input specs
        self.masks = {}
        for key, border_kwargs in borders.items():
            spec = specs.obs.get(key, None)
            if spec is None or not isinstance(spec, CameraSpec):
                raise ValueError(f"Camera spec for key '{key}' not found in specs. Got {spec}")

            hws = [stream.height_width for stream in spec.streams.values()]
            if not all(hw == hws[0] for hw in hws):
                raise ValueError(
                    f"All camera streams must have the same height and width, but got {hws}"
                )

            border = TriangleCropBorder(**border_kwargs)
            if not border.in_img(*hws[0]):
                raise ValueError(f"Border for {key} is out of image bounds: {border} vs {hws[0]}")

            self.masks[key] = border.build_mask(*hws[0])

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        for key, mask in self.masks.items():
            if key not in tensordict["obs"]:
                raise KeyError(f"Key '{key}' not found in tensordict. Available keys: {list(tensordict.keys())}")

            spec = self.specs.obs[key]
            assert isinstance(spec, CameraSpec)
            images = tensordict["obs", key]

            for name, stream in spec.streams.items():
                assert isinstance(images[name], torch.Tensor)
                if isinstance(stream, PointMapStream):
                    continue
                
                if stream.channel_order == "HWC":
                    mask_stacked = mask.view((1,) * (images[name].ndim - 3) + mask.shape + (1,))
                else:  # "CHW", "HW"
                    mask_stacked = mask.view((1,) * (images[name].ndim - 2) + mask.shape)

                mask_stacked = mask_stacked.expand_as(images[name]).to(images[name].device)
                mask_val = torch.tensor(self.mask_value, dtype=images[name].dtype, device=images[name].device)
                images[name] = torch.where(mask_stacked, images[name], mask_val)

            tensordict["obs", key] = images

        return tensordict
