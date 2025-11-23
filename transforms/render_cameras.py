from __future__ import annotations

import multiprocessing as mp
import traceback
from typing import Sequence

from tensordict import TensorDict

from environments.specs import (
    CameraSpec,
    DataSpecs,
    DepthStream,
    ImageStream,
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

STREAM_TYPES = {
    "rgb": RGBStream,
    "depth": DepthStream,
    "pointmap": PointMapStream,
    "ir": ImageStream,
}


class RenderCameras(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        stream_types: str | Sequence[str] | None = None,
        stream_names: str | Sequence[str] | None = None,
        min_depth: float | None = None,
        max_depth: float | None = None,
        depth_colormap: str = "magma",
        show_stream_names: bool = True,
    ) -> None:
        # TODO: extend tile_images to specify a particular coordinate for each
        # stream, e.g. each camera is its own column/row but they may have
        # different numbers of streams

        input_specs = {
            key: {name: (spec, stream) for name, stream in spec.streams.items()}
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
        }

        if stream_types is not None:
            if isinstance(stream_types, str):
                stream_types = [stream_types]

            stream_classes = tuple(
                STREAM_TYPES[stream_type.lower()] for stream_type in stream_types
            )
            input_specs = {
                key: {
                    name: (spec, stream)
                    for name, (spec, stream) in streams.items()
                    # exact type match because ImageStream is the base class for others
                    if type(stream) in stream_classes
                }
                for key, streams in input_specs.items()
            }

        if stream_names is not None:
            if isinstance(stream_names, str):
                stream_names = [stream_names]

            input_specs = {
                key: {
                    name: (spec, stream)
                    for name, (spec, stream) in streams.items()
                    if name in stream_names
                }
                for key, streams in input_specs.items()
            }

        height_widths = [
            stream.height_width
            for streams in input_specs.values()
            for (spec, stream) in streams.values()
        ]
        if not height_widths:
            raise ValueError(
                f"No matching image streams found for with types {stream_types} and names {stream_names}."
            )

        if not all(hw == height_widths[0] for hw in height_widths):
            raise ValueError(
                f"All camera streams must have the same height and width, but got {height_widths}"
            )

        streams_per_camera = [len(streams) for streams in input_specs.values()]
        if not all(n == streams_per_camera[0] for n in streams_per_camera):
            streams_per_camera = {
                key: len(streams) for key, streams in input_specs.items()
            }
            raise ValueError(
                f"All cameras must have the same number of selected streams, but got {streams_per_camera}"
            )

        self.input_specs = input_specs
        height, width = height_widths[0]
        self.width = width
        self.height = height
        # self.tiled_height, self.tiled_width = find_tiling(n_images)
        self.tiled_width = len(input_specs)  # number of cameras
        self.tiled_height = len(height_widths) // self.tiled_width

        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_colormap = depth_colormap
        self._specs = specs
        self.show_stream_names = show_stream_names

        ctx = mp.get_context("spawn")

        child_end, self.pipe = ctx.Pipe(duplex=False)
        self.process = ctx.Process(
            target=rendering_process,
            args=(child_end,),
            kwargs={
                "window_title": "Camera Views",
                "tile_width": self.width,
                "tile_height": self.height,
                "n_tiles_width": self.tiled_width,
                "n_tiles_height": self.tiled_height,
            },
            daemon=True,  # daemon so it doesn't outlive the parent if something goes wrong
        )
        self.process.start()

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        images = []
        stream_names = []

        for key, streams in self.input_specs.items():
            for name, (spec, stream) in streams.items():
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

        try:
            self.pipe.send(
                {
                    "image": image,
                    "stream_names": stream_names if self.show_stream_names else [],
                }
            )
        except (BrokenPipeError, EOFError):
            # rendering process has crashed or exited
            raise KeyboardInterrupt("Pygame quit")

        return tensordict

    def close(self) -> None:
        self.pipe.send("QUIT")
        self.process.terminate()
        self.process.join()
        self.pipe.close()


def rendering_process(
    pipe_end: mp.connection.Connection,
    tile_width: int,
    tile_height: int,
    n_tiles_width: int,
    n_tiles_height: int,
    window_title: str = "Camera Views",
):
    import pygame

    pygame.init()

    screen = pygame.display.set_mode(
        (n_tiles_width * tile_width, n_tiles_height * tile_height)
    )
    pygame.display.set_caption(window_title)
    screen.fill((0, 0, 0))  # Clear the screen

    # Initialize font for labels
    pygame.font.init()
    font = pygame.font.Font(None, 24)

    try:
        while True:
            # Non-blocking check for incoming messages
            if pipe_end.poll():
                message = pipe_end.recv()
                if message == "QUIT":
                    break
                elif isinstance(message, dict):
                    image = message["image"]
                    stream_names = message.get("stream_names", [])

                    surface = pygame.surfarray.make_surface(image.transpose(1, 0, 2))
                    screen.blit(surface, (0, 0))

                    # Add labels to each camera view
                    for i, label in enumerate(stream_names):
                        row = i % n_tiles_height
                        col = i // n_tiles_height
                        x = col * tile_width + 10
                        y = row * tile_height + 10

                        # Create text surface with black background for better visibility
                        text_surface = font.render(label, True, (255, 255, 255))
                        text_rect = text_surface.get_rect()

                        # Create background rectangle
                        bg_rect = pygame.Rect(
                            x - 5, y - 2, text_rect.width + 10, text_rect.height + 4
                        )
                        pygame.draw.rect(screen, (0, 0, 0, 180), bg_rect)

                        # Blit text
                        screen.blit(text_surface, (x, y))

                    pygame.display.flip()  # Update display

            # Keep the window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    raise KeyboardInterrupt("Pygame quit")

    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        try:
            pygame.font.quit()
            pygame.quit()
        except Exception:
            pass

        pipe_end.close()
