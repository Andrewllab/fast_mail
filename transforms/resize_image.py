from __future__ import annotations

import dataclasses

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from environments.specs import CameraSpec, DataSpecs
from transforms.base_transform import KeyMapping, Transform


class ResizeImage(Transform):
    def __init__(
        self,
        specs: DataSpecs,
        factor: float | None = None,
        shape: int | tuple[int, int] | None = None,
        scale_depth_images: bool = False,
        interpolation: InterpolationMode | None = InterpolationMode.BILINEAR,
        depth_interpolation: InterpolationMode | None = InterpolationMode.NEAREST_EXACT,
        antialias: bool = True,
    ) -> None:

        if factor is None and shape is None:
            raise ValueError("Either factor or shape must be provided.")
        elif factor is not None and shape is not None:
            raise ValueError("Only one of factor or shape may be provided.")

        interpolation = (
            interpolation if interpolation is not None else InterpolationMode.BILINEAR
        )
        depth_interpolation = (
            depth_interpolation
            if depth_interpolation is not None
            else InterpolationMode.NEAREST_EXACT
        )
        self.antialias = antialias

        # do we resize all images, or only rgb images?
        predicate = lambda spec: (
            isinstance(spec, CameraSpec)
            if scale_depth_images
            else lambda spec: type(spec) is CameraSpec
        )
        camera_specs = {key: spec for key, spec in specs.obs.items() if predicate(spec)}

        # we will create a dictionary of kwargs for resize() for each camera image
        self.cam_kwargs = {}
        obs_specs = dict(specs.obs)  # copy obs specs for local modification

        for key, spec in camera_specs.items():
            assert isinstance(spec, CameraSpec)

            # compute shape of the resized image
            if factor is not None:
                new_shape = (
                    int(factor * spec.shape[-2]),
                    int(factor * spec.shape[-1]),
                )
            elif isinstance(shape, int):
                # if shape is an int, it is the new size for the shortest side,
                # and the longest side is scaled to maintain the aspect ratio
                shortest_side = min(spec.shape[-2:])
                new_shape = (
                    int(shape * spec.shape[-2] / shortest_side),
                    int(shape * spec.shape[-1] / shortest_side),
                )
            elif isinstance(shape, tuple):
                # if shape is a tuple, it is the new shape
                new_shape = shape
            else:
                raise ValueError("Shape must be an int or a tuple of ints, or None.")

            # save the kwargs for resize()
            self.cam_kwargs[key] = {
                "size": new_shape,
                "interpolation": (
                    depth_interpolation
                    if type(spec) is not CameraSpec
                    else interpolation
                ),
            }

            # update specs with resized shape and modified camera intrinsics
            spec = dataclasses.replace(
                spec,
                shape=(spec.shape[:-2] + new_shape),
            )
            if spec.intrinsics is not None:
                spec = dataclasses.replace(
                    spec,
                    intrinsics=spec.intrinsics.resize(new_shape),
                )
            obs_specs[key] = spec

        self._specs = specs.replace(obs=obs_specs)

    @property
    def key_mappings(self) -> list[KeyMapping]:
        # we take all images at once and write them back to their sources
        # we cannot process the images one by one, because then we don't know
        # what kind of image we are dealing with, e.g. rgb or depth
        return [
            KeyMapping(
                in_keys=[("obs", key) for key in self.cam_kwargs],
                out_keys=[("obs", key) for key in self.cam_kwargs],
            )
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def __call__(self, *imgs: torch.Tensor) -> tuple[torch.Tensor]:

        resized_imgs = []
        for img, cam_cfg in zip(imgs, self.cam_kwargs.values()):
            resized_imgs.append(F.resize(img, **cam_cfg, antialias=self.antialias))

        return tuple(resized_imgs)
