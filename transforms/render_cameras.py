from __future__ import annotations

from typing import Sequence

import pygame
from tensordict import TensorDict

from environments.specs import (
    CameraSpec,
    DataSpecs,
    DepthStream,
    PointMapStream,
    RGBStream,
)
from transforms.base_transform import Transform
from utils.rendering import (
    depth_to_renderable,
    find_tiling,
    intensity_to_renderable,
    rgb_to_renderable,
    tile_images,
)


class RenderCameras(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        stream_names: str | Sequence[str] | None = None,
        min_depth: float | None = None,
        max_depth: float | None = None,
        depth_colormap: str = "magma",
        show_stream_names: bool = True,
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
            if stream_names is None
            or name in stream_names
            and not isinstance(stream, PointMapStream)
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
        pygame.display.set_caption("Camera Views")
        self.screen.fill((0, 0, 0))  # Clear the screen

        # Initialize font for labels
        pygame.font.init()
        self.font = pygame.font.Font(None, 24)

        self.width = width
        self.height = height

        self._input_specs = input_specs
        self.stream_names = stream_names
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_colormap = depth_colormap
        self._specs = specs
        self.show_stream_names = show_stream_names

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise KeyboardInterrupt("Pygame quit")

        images = []
        stream_names = []
        for key, spec in self._input_specs.items():
            for name, stream in spec.streams.items():
                if self.stream_names is not None and name not in self.stream_names:
                    continue
                elif isinstance(stream, PointMapStream):
                    continue

                image = tensordict["obs", key, name]
                image = image[0, -1].cpu().numpy()  # remove batch and time dimensions
                if isinstance(stream, RGBStream):
                    image = rgb_to_renderable(image, stream.channel_order)
                elif isinstance(stream, DepthStream):
                    image = depth_to_renderable(
                        image,
                        depth_min=self.min_depth,
                        depth_max=self.max_depth,
                        colormap_name=self.depth_colormap,
                    )
                else:
                    assert stream.channels is None
                    image = intensity_to_renderable(image)

                images.append(image)
                stream_names.append(f"{key}/{name}")

        image = tile_images(images, self.tiled_height, self.tiled_width, vertical=True)

        surface = pygame.surfarray.make_surface(image.transpose(1, 0, 2))
        self.screen.blit(surface, (0, 0))

        if self.show_stream_names:
        # Add labels to each camera view
            for i, label in enumerate(stream_names):
                row = i % self.tiled_height
                col = i // self.tiled_height
                x = col * self.width + 10
                y = row * self.height + 10

                # Create text surface with black background for better visibility
                text_surface = self.font.render(label, True, (255, 255, 255))
                text_rect = text_surface.get_rect()

                # Create background rectangle
                bg_rect = pygame.Rect(
                    x - 5, y - 2, text_rect.width + 10, text_rect.height + 4
                )
                pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect)

                # Blit text
                self.screen.blit(text_surface, (x, y))

        pygame.display.flip()  # Update display

        return tensordict

    def close(self) -> None:
        pygame.font.quit()
        pygame.quit()
