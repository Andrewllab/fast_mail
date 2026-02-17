from typing import Literal, Sequence

import matplotlib as mpl
import numpy as np


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


def tile_images(
    images: Sequence[np.ndarray],
    tiled_height: int,
    tiled_width: int,
    vertical: bool = False,
) -> np.ndarray:
    height, width, n_channels = images[0].shape

    tiled_frame_HhWwN = np.zeros(
        (tiled_height * height, tiled_width * width, n_channels), dtype=np.uint8
    )

    # this array shares memory with the original array, but is more intuitive
    # for writing to
    tiled_frame_HWhwN = tiled_frame_HhWwN.reshape(
        (tiled_height, height, tiled_width, width, n_channels)
    ).transpose(0, 2, 1, 3, 4)

    images_it = iter(images)
    if vertical:
        # tile images top to bottom, left to right
        for j in range(tiled_width):
            for i in range(tiled_height):
                try:
                    tiled_frame_HWhwN[i, j] = next(images_it)
                except StopIteration:
                    return tiled_frame_HhWwN
    else:
        # tile images left to right, top to bottom
        for i in range(tiled_height):
            for j in range(tiled_width):
                try:
                    tiled_frame_HWhwN[i, j] = next(images_it)
                except StopIteration:
                    return tiled_frame_HhWwN

    return tiled_frame_HhWwN


def depth_to_renderable(
    depth: np.ndarray,
    depth_min: float | None = None,
    depth_max: float | None = None,
    colormap_name: str = "magma",
    invert_colormap: bool = False,
) -> np.ndarray:
    """Convert a depth image to a renderable rgb array.

    The depth image must be of shape (H, W) (no batching) with a float dtype.
    Invalid values are replaced with 0.0. The depth values are scaled to [0, 1],
    either using the provided `depth_min` and `depth_max` or the min and max
    values of the depth image.

    Args:
        depth (np.ndarray): Depth image of shape (H, W).
        depth_min (float | None): Minimum depth value for scaling. Default is
            to use empirical min.
        depth_max (float | None): Maximum depth value for scaling. Default is
            to use empirical max.
        colormap_name (str): Name of the colormap to use. Defaults to "magma".
    """
    assert (
        depth.ndim == 2
    ), f"Depth image must have 2 dimensions, got {depth.ndim} instead."
    assert np.issubdtype(
        depth.dtype, np.floating
    ), f"Depth image must be of float type, got {depth.dtype} instead."

    # handle invalid values
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize to [0, 1]
    depth_min = depth_min if depth_min is not None else depth.min()
    depth_max = depth_max if depth_max is not None else depth.max()
    depth_range = depth_max - depth_min + 1e-6  # avoid division by zero
    depth = (depth - depth_min) / depth_range
    depth = depth.clip(0, 1)  # ensure depth is in [0, 1]

    if invert_colormap:
        depth = 1 - depth  # invert the colormap

    # Apply colormap
    cmap = mpl.colormaps[colormap_name]
    colored = cmap(depth)[:, :, :3]  # Drop alpha channel → shape (H, W, 3)

    # Convert to 8-bit RGB
    rgb = (colored * 255).clip(min=0, max=255).astype(np.uint8)

    return rgb


def intensity_to_renderable(intensity: np.ndarray) -> np.ndarray:
    """Convert an intensity image to a renderable rgb array by repeating the
    intensity value across three channels.

    The intensity image must be of shape (H, W) (no batching) with dtype uint8,
    where 255 is white and 0 is black.

    Args:
        intensity (np.ndarray): Intensity image of shape (H, W).
    """
    assert (
        intensity.ndim == 2
    ), f"Intensity image must have 2 dimensions, got {intensity.ndim} instead."
    assert (
        intensity.dtype == np.uint8
    ), f"Intensity image must be of uint8 type, got {intensity.dtype} instead."

    # add a channel dimension and repeat the intensity value
    # to create a 3-channel image
    rgb = np.expand_dims(intensity, axis=-1).repeat(repeats=3, axis=-1)

    return rgb


def rgb_to_renderable(
    rgb: np.ndarray, channel_order: Literal["HWC", "CHW"]
) -> np.ndarray:
    """Convert an rgb image to a renderable rgb array.

    The intensity image must be of shape (C, H, W) or (H, W, C) (no batching)
    with dtype uint8.

    Args:
        intensity (np.ndarray): Intensity image of shape (H, W).
    """

    """Convert a torch tensor to a numpy array.
    The tensor is assumed to be in the format (C, H, W) or (H, W, C).
    """
    assert rgb.ndim == 3, f"RGB image must have 3 dimensions, got {rgb.ndim} instead."

    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).clip(min=0, max=255).astype(np.uint8)

    if channel_order == "CHW":
        rgb = np.moveaxis(rgb, 0, -1)  # move channel to last dimension

    assert (
        rgb.shape[-1] == 3
    ), f"RGB image must have 3 channels, got {rgb.shape[-1]} instead."

    return rgb


def to_rgb_color(color_code: str) -> list[int]:
    assert color_code[0] == "#"
    assert len(color_code) == 7
    return [
        int(color_code[1:3], 16),
        int(color_code[3:5], 16),
        int(color_code[5:7], 16),
    ]
