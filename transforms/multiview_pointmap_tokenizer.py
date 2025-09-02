from __future__ import annotations

import logging
from typing import Callable, Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from environments.specs import (
    CameraSpec,
    DataSpecs,
    EmbedSpec,
    PointMapStream,
    RGBStream,
)
from transforms.base_transform import Transform
from utils.nested import cat_nested

log = logging.getLogger(__name__)


class MultiviewPointMapTokenizer(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        pointmap_encoder: Callable[[int, int], nn.Module],
        embed_dim: int,
        image_encoder: Callable[[int, int], nn.Module] | None = None,
        fusion_type: Literal["6ch", "add", "cat"] | None = None,
        # shared_encoder: bool = False,  # TODO: implement this
    ):
        super().__init__()

        if fusion_type == "6ch":
            stream_types = (PointMapStream,)
            self.late_fusion = False
            in_channels = 6
        elif fusion_type is None:
            stream_types = (PointMapStream,)
            self.late_fusion = False
            in_channels = 3
        elif fusion_type in ["add", "cat"]:
            stream_types = (RGBStream, PointMapStream)
            self.late_fusion = True
            in_channels = 3
            rgb_channels = 3
            if image_encoder is None:
                raise ValueError(
                    "`image_encoder` must be provided for add and cat pointmap fusion types"
                )
        else:
            raise ValueError(
                f"pointmap_type must be one of '6ch', 'add', or 'cat', but got {fusion_type}"
            )
        self.fusion_type = fusion_type
        self.stream_types = stream_types

        input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, CameraSpec)
            # all required stream types must be present in the spec
            and all(
                any(isinstance(stream, stream_type) for stream in spec.streams.values())
                for stream_type in stream_types
            )
        }
        if not input_specs:
            raise ValueError("No valid input specs found.")
        self._input_specs = input_specs

        input_streams = [
            stream
            for spec in input_specs.values()
            for stream in spec.streams.values()
            if isinstance(stream, stream_types)
        ]

        # verify that all streams have the same time dimension
        if not all(stream.time == input_streams[0].time for stream in input_streams):
            raise ValueError(
                "All input streams must have the same time dimension."
                f"Got {[stream.time for stream in input_streams]}"
            )

        # verify that all streams have the same resolution
        if not all(
            stream.height_width == input_streams[0].height_width
            for stream in input_streams
        ):
            raise ValueError(
                "All input streams must have the same height and width when using a shared encoder."
                f"Got {[stream.height_width for stream in input_streams]}"
            )

        # verify that all point map streams have the correct number of channels
        if not all(
            stream.channels == in_channels
            for stream in input_streams
            if isinstance(stream, PointMapStream)
        ):
            raise ValueError(
                f"All input streams must have {in_channels} channels when using {fusion_type} fusion."
                f"Got {[stream.channels for stream in input_streams if isinstance(stream, PointMapStream)]}"
            )

        # instantiate rgb model(s)
        self.pointmap_model = pointmap_encoder(in_channels, embed_dim)
        if self.late_fusion:
            assert image_encoder is not None
            self.image_model = image_encoder(rgb_channels, embed_dim)

        # each camera produces one token
        # if we have stereo rgb, we just take the left camera
        # we have 2 tokens per view with "cat" fusion mode, otherwise only 1 per view
        n_tokens = (
            2 * len(input_specs) if self.fusion_type == "cat" else len(input_specs)
        )
        new_spec = EmbedSpec(embed_dim=embed_dim, n_tokens=n_tokens)

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        if "embed" in obs_specs:
            embed_spec = obs_specs["embed"]
            assert isinstance(embed_spec, EmbedSpec)
            new_spec = embed_spec.concat(new_spec)
        obs_specs["embed"] = new_spec
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def forward(self, tensordict: TensorDict) -> TensorDict:
        default_float_dtype = torch.get_default_dtype()

        pointmap_streams = []
        rgb_streams = []
        for key, spec in self._input_specs.items():
            for name, stream in spec.streams.items():
                if not isinstance(stream, self.stream_types):
                    continue

                image = tensordict["obs", key, name]

                if stream.channel_order == "HWC":
                    image = torch.movedim(image, -1, -3)
                elif stream.channel_order == "HW":
                    image = torch.unsqueeze(image, dim=-3)

                assert (
                    image.shape[-2:] == stream.height_width
                ), f"Expected image shape {stream.height_width}, but got {image.shape[-2:]} for {key}/{name}"

                if image.dtype == torch.uint8:
                    image = image.to(dtype=default_float_dtype).div(255)

                if isinstance(stream, PointMapStream):
                    pointmap_streams.append(image)
                elif isinstance(stream, RGBStream):
                    rgb_streams.append(image)

        # we stack and flatten rather than concatenate, to keep images from the same time step together
        # (B, T, C, H, W) -> (B, T, N, C, H, W)
        point_maps = torch.stack(pointmap_streams, dim=-4)

        leading_dims, shape = (point_maps.shape[:3], point_maps.shape[-3:])
        # (B, T, N, C, H, W) -> (B*T*N, C, H, W)
        point_maps = point_maps.view(-1, *shape)

        # (B*T*N, C, H, W) -> (B*T*N, D)
        features = self.pointmap_model(point_maps)
        # keep time and number of cameras flattened
        # (B*T*N, D) -> (B, T, N, D)
        features = torch.unflatten(features, dim=0, sizes=leading_dims)

        if self.late_fusion:
            # Do the same for rgb images
            rgb_images = torch.stack(rgb_streams, dim=-4)
            assert shape == rgb_images.shape[-3:]

            rgb_images = rgb_images.view(-1, *shape)
            features_rgb = self.image_model(rgb_images)
            features_rgb = torch.unflatten(features_rgb, dim=0, sizes=leading_dims)

            # features: (B, T, N, D)
            if self.fusion_type == "cat":
                # concatenate along N dimension
                features = torch.cat([features, features_rgb], dim=-2)
            else:
                assert self.fusion_type == "add"
                features = features + features_rgb

        # (B, T, N, D) -> (B, T*N, D)
        features = features.flatten(start_dim=1, end_dim=2)

        if (obs_embed := tensordict["obs"].get("embed")) is not None:
            # concatenate along N dimension of embedding
            # obs_embed: (B, N, D)
            features = cat_nested([obs_embed, features], dim=-2)

        tensordict["obs", "embed"] = features

        return tensordict
