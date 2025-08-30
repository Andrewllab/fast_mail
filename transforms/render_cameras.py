from __future__ import annotations

import itertools
from typing import Sequence

import matplotlib.cm as cm
import numpy as np
import pygame
import torch
from tensordict import TensorDict

from environments.specs import (
    CameraSpec,
    ChannelOrderType,
    DataSpecs,
    DepthStream,
    RGBStream,
)
from transforms.base_transform import Transform


class RenderCameras(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        stream_names: str | Sequence[str] | None = None,
        min_depth: float | None = None,
        max_depth: float | None = None,
        depth_colormap: str = "magma",
    ) -> None:
        input_specs = {
            key: spec for key, spec in specs.obs.items() if isinstance(spec, CameraSpec)
        }

        if isinstance(stream_names, str):
            stream_names = [stream_names]

        input_streams = [
            stream
            for spec in input_specs.values()
            for name, stream in spec.streams.items()
            if stream_names is None or name in stream_names
        ]

        n_images = len(input_streams)
        if n_images == 0:
            raise ValueError("No camera specs found.")

        height_widths = [stream.height_width for stream in input_streams]
        if not all(hw == height_widths[0] for hw in height_widths):
            raise ValueError(
                f"All camera streams must have the same height and width, but got {height_widths}"
            )

        height, width = height_widths[0]
        self.tiled_height, self.tiled_width = find_tiling(n_images)

        self.screen = pygame.display.set_mode(
            (width * self.tiled_width, height * self.tiled_height)
        )
        # pygame.display.set_caption(f"obs.{key}")
        self.screen.fill((0, 0, 0))  # Clear the screen

        self._input_specs = input_specs
        self.stream_names = stream_names
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_colormap = depth_colormap
        self._specs = specs

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise KeyboardInterrupt("Pygame quit")

        images = []
        for key, spec in self._input_specs.items():
            for name, stream in spec.streams.items():
                if self.stream_names is not None and name not in self.stream_names:
                    continue

                image = tensordict["obs", key, name]
                image = image[0, -1]  # remove batch and time dimensions
                if isinstance(stream, RGBStream):
                    image = rgb_tensor_to_np(image, stream.channel_order)
                elif isinstance(stream, DepthStream):
                    image = depth_tensor_to_np(
                        image,
                        depth_min=self.min_depth,
                        depth_max=self.max_depth,
                        colormap_name=self.depth_colormap,
                    )
                else:
                    assert stream.channels is None
                    image = intensity_tensor_to_np(image)

                images.append(image)

        image = tile_images(images, self.tiled_height, self.tiled_width, vertical=True)

        surface = pygame.surfarray.make_surface(image.transpose(1, 0, 2))
        self.screen.blit(surface, (0, 0))

        pygame.display.flip()  # Update display

        return tensordict

    def close(self) -> None:
        pygame.quit()


def find_tiling(n_images: int) -> tuple[int, int]:
    """Find a tiling to represent `n_images` images in one big PxQ tiling.
    P and Q are chosen to be factors of n_images, as long as this is possible
    with an aspect ratio between 1:1 and 2:1. Otherwise P and Q are chosen to
    be as close to each other as possible.
    """
    # first, try to factorize n_images with an aspect ratio between 1:1
    # and 2:1
    max_tiled_height = int(np.floor(np.sqrt(n_images)))
    min_tiled_height = int(np.ceil(np.sqrt(n_images / 2)))
    try:
        tiled_height = next(
            i
            for i in range(max_tiled_height, min_tiled_height - 1, -1)
            if n_images % i == 0
        )
        tiled_width = n_images // tiled_height
        return (tiled_height, tiled_width)

    except StopIteration:
        pass

    # if such factors do not exist, construct a grid that is roughly
    # square. Additional tiles will be filled in with black
    tiled_height = int(np.ceil(np.sqrt(n_images)))
    tiled_width = int(np.ceil(float(n_images) / tiled_height))
    return (tiled_height, tiled_width)


def rgb_tensor_to_np(
    image: torch.Tensor, channel_order: ChannelOrderType
) -> np.ndarray:
    """Convert a torch tensor to a numpy array.
    The tensor is assumed to be in the format (C, H, W) or (H, W, C).
    """
    if channel_order == "CHW":
        image = torch.movedim(image, 0, -1)  # move channel to last dimension

    # convert back to uint8 if needed
    if image.dtype != torch.uint8:
        image = image.mul(255).clamp(0, 255).to(torch.uint8)

    return image.cpu().numpy()


def intensity_tensor_to_np(image: torch.Tensor) -> np.ndarray:
    # convert back to uint8 if needed
    if image.dtype != torch.uint8:
        image = image.mul(255).clamp(0, 255).to(torch.uint8)

    # add a channel dimension and repeat the intensity value
    # to create a 3-channel image
    image = image.unsqueeze(-1).expand(-1, -1, 3)

    return image.cpu().numpy()


def depth_tensor_to_np(
    depth: torch.Tensor,
    depth_min: float | None = None,
    depth_max: float | None = None,
    colormap_name="magma",
) -> np.ndarray:
    """
    Convert a torch tensor to a numpy array.
    The tensor is assumed to be in the format (H, W).
    """
    # handle invalid values
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to [0, 1]
    if depth_min is None:
        depth_min = depth.min()
    if depth_max is None:
        depth_max = depth.max()
    depth_range = depth_max - depth_min + 1e-6  # avoid division by zero
    depth = (depth - depth_min) / depth_range

    depth_np = depth.cpu().numpy()

    # Apply colormap
    cmap = cm.get_cmap(colormap_name)
    colored = cmap(depth_np)[:, :, :3]  # Drop alpha channel → shape (H, W, 3)

    # Convert to 8-bit RGB
    rgb = (colored * 255).clip(min=0, max=255).astype(np.uint8)

    return rgb


def tile_images(
    images: Sequence[np.ndarray],
    tiled_height: int,
    tiled_width: int,
    vertical: bool = False,
) -> np.ndarray:
    n_images = len(images)
    height, width, n_channels = images[0].shape

    tiled_frame_HhWwN = np.zeros(
        (tiled_height * height, tiled_width * width, n_channels), dtype=np.uint8
    )

    # this array shares memory with the original array, but is more intuitive
    # for writing to
    tiled_frame_HWhwN = tiled_frame_HhWwN.reshape(
        (tiled_height, height, tiled_width, width, n_channels)
    ).transpose(0, 2, 1, 3, 4)

    n = 0
    if vertical:
        # tile images top to bottom, left to right
        iterators = (range(tiled_width), range(tiled_height))
    else:
        # tile images left to right, top to bottom
        iterators = (range(tiled_height), range(tiled_width))

    for i, j in itertools.product(*iterators):
        if n >= n_images:
            return tiled_frame_HhWwN

        if vertical:
            i, j = j, i

        tiled_frame_HWhwN[i, j] = images[n]
        n += 1

    return tiled_frame_HhWwN
