from __future__ import annotations

import multiprocessing as mp
import traceback
from pathlib import Path

import numpy as np
from tensordict import TensorDict

from environments.specs import DataSpecs, DepthStream, RGBStream
from transforms.base_transform import Transform
from utils.paths import resolve_path
from utils.rendering import (
    depth_to_renderable,
    intensity_to_renderable,
    rgb_to_renderable,
)


class RenderImageDiff(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        image_path: str,
        cam_name: str,
        stream_name: str = "left",
    ) -> None:
        inputs = (
            (cam_name, stream_name),
            (spec := specs.obs[cam_name], stream := spec.streams[stream_name]),
        )

        self.height, self.width = stream.height_width
        self.inputs = inputs
        self.image_path = resolve_path(image_path)
        self._specs = specs

        ctx = mp.get_context("spawn")

        child_end, self.pipe = ctx.Pipe(duplex=False)
        self.process = ctx.Process(
            target=rendering_process,
            args=(child_end,),
            kwargs={
                "image_path": self.image_path,
                "tile_width": self.width,
                "tile_height": self.height,
                "window_title": f"{cam_name}/{stream_name}",
            },
            daemon=True,  # daemon so it doesn't outlive the parent if something goes wrong
        )
        self.process.start()

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, tensordict: TensorDict) -> TensorDict:
        (key, name), (spec, stream) = self.inputs

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

        try:
            self.pipe.send({"image": image})
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
    image_path: Path,
    tile_width: int,
    tile_height: int,
    window_title: str,
):
    import pygame

    pygame.init()

    # image, reference, and difference will be shown side by side
    screen = pygame.display.set_mode((3 * tile_width, tile_height))
    pygame.display.set_caption(window_title)
    screen.fill((0, 0, 0))  # Clear the screen

    # --- Load PNG image ---
    ref_surface = pygame.image.load(str(image_path)).convert_alpha()

    # scale PNG to fit space
    ref_surface = pygame.transform.scale(ref_surface, (tile_width, tile_height))

    # get PNG image as numpy array and transpose to (H, W, C)
    ref_image = pygame.surfarray.array3d(ref_surface).transpose(1, 0, 2)
    ref_image_float = ref_image.astype(np.float32)

    try:
        while True:
            # Non-blocking check for incoming messages
            if pipe_end.poll():
                message = pipe_end.recv()
                if message == "QUIT":
                    break

                elif isinstance(message, dict):
                    image = message["image"]

                    # Clear background
                    screen.fill((30, 30, 30))

                    # Blit rendered image onto left section
                    surface = pygame.surfarray.make_surface(image.transpose(1, 0, 2))
                    screen.blit(surface, (0, 0))

                    # Blit PNG image onto middle section
                    screen.blit(ref_surface, (tile_width, 0))

                    # Compute difference image
                    # convert to float32 to avoid subtraction with unsigned ints
                    diff_image = image.astype(np.float32) - ref_image_float

                    # Convert to grayscale
                    diff_image = diff_image.mean(axis=-1, keepdims=True)
                    diff_image = np.repeat(diff_image, 3, axis=-1)

                    # rescale from [-255, 255] to [0, 255] so that no
                    # difference maps to grey (127.5)
                    diff_image = diff_image / 2 + 127.5

                    # convert back to uint8 and render
                    diff_image = diff_image.clip(0, 255).astype("uint8")
                    diff_surface = pygame.surfarray.make_surface(
                        diff_image.transpose(1, 0, 2)
                    )
                    screen.blit(diff_surface, (2 * tile_width, 0))

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
