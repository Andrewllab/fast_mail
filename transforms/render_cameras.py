import numpy as np
import pygame
import torch

from environments.specs import DataSpecs, IntensityCameraSpec
from transforms.base_transform import KeyMapping, Transform


class RenderCameras(Transform):
    def __init__(self, specs: DataSpecs) -> None:
        # TODO: add support for multiple cameras
        # TODO: check for channel ordering

        rgb_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, IntensityCameraSpec)
        }

        self.rgb_key = list(rgb_specs.keys())[0]
        self.rgb_spec = rgb_specs[self.rgb_key]
        self.rgb_shape = self.rgb_spec.shape
        height, width = self.rgb_shape[-3:-1]

        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(f"obs.{self.rgb_key}")
        self.screen.fill((0, 0, 0))  # Clear the screen

        self._output_specs = specs

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [KeyMapping(in_keys=[("obs", self.rgb_key)], out_keys=["_"])]

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def __call__(self, image: torch.Tensor) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise KeyboardInterrupt("Pygame quit")

        image = image.squeeze(0).squeeze(0)  # remove batch and time dimensions
        surface = pygame.surfarray.make_surface(np.rot90(image.cpu().numpy()))
        self.screen.blit(surface, (0, 0))

        pygame.display.flip()  # Update display

    def close(self) -> None:
        pygame.quit()
