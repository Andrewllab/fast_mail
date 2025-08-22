from __future__ import annotations

import logging
from typing import Callable, Literal, Optional

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
        pointmap_encoder: Callable[[], nn.Module],
        embed_dim: int,
        fusion_type: Literal["6ch", "add", "cat"] = "6ch",
        image_encoder: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()

        if fusion_type == "6ch":
            stream_types = (PointMapStream,)
            self.late_fusion = False
        elif fusion_type in ["add", "cat"]:
            stream_types = (RGBStream, PointMapStream)
            self.late_fusion = True
            assert image_encoder is not None, "image_encoder must be provided for add and cat pointmap types"
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
        self._input_specs = input_specs

        input_streams = [
            stream
            for spec in input_specs.values()
            for stream in spec.streams.values()
            if isinstance(stream, stream_types)
        ]

        if not all(stream.time == input_streams[0].time for stream in input_streams):
            raise ValueError(
                "All input streams must have the same time dimension."
                f"Got {[stream.time for stream in input_streams]}"
            )

        # instantiate rgb model(s)
        if not all(
            input_stream.height_width == input_streams[0].height_width
            for input_stream in input_streams
        ):
            raise ValueError(
                "All input streams must have the same height and width when using a shared encoder."
                f"Got {[stream.height_width for stream in input_streams]}"
            )

        self.pointmap_model = pointmap_encoder()
        self.image_model = image_encoder() if image_encoder else None

        # each camera produces one token
        # if we have stereo rgb, we just take the left camera
        embed_dim_ = 2 * embed_dim if self.fusion_type == "cat" else embed_dim
        n_tokens = len(input_specs) if not self.late_fusion else len(input_specs) // 2
        new_spec = EmbedSpec(embed_dim=embed_dim_, n_tokens=n_tokens)

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
    
    def late_fusion_fn(self, features_pm: torch.Tensor, features_rgb: torch.Tensor) -> torch.Tensor:
        if self.fusion_type == "add":
            return features_pm + features_rgb
        elif self.fusion_type == "cat":
            return torch.cat([features_pm, features_rgb], dim=-1)
        else:
            raise ValueError(f"Unknown pointmap type: {self.fusion_type}")

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

            if image.dtype == torch.uint8:
                image = image.to(dtype=default_float_dtype).div(255)
                
            if isinstance(stream, PointMapStream):
                pointmap_streams.append(image)
            elif isinstance(stream, RGBStream):
                rgb_streams.append(image)

        # we stack and flatten rather than concatenate, to keep images from the same time step together
        # (B, T, C, H, W) -> (B, T, N, C, H, W)
        point_maps = torch.stack(pointmap_streams, dim=-4)

        B, shape = (point_maps.shape[0], point_maps.shape[-3:])
        # (B, T, N, C, H, W) -> (B*T*N, C, H, W)
        point_maps = point_maps.view(-1, *shape)

        # (B*T*N, C, H, W) -> (B*T*N, D)
        features = self.pointmap_model(point_maps)
        # keep time and number of cameras flattened
        # (B*T*N, D) -> (B, T*N, D)
        features = torch.unflatten(features, dim=0, sizes=(B, -1))
        
        if self.image_model is not None:
            # Do the same for rgb images
            rgb_images = torch.stack(rgb_streams, dim=-4)
            assert shape == rgb_images.shape[-3:]

            rgb_images = rgb_images.view(-1, *shape)
            features_rgb = self.image_model(rgb_images)
            features_rgb = torch.unflatten(features_rgb, dim=0, sizes=(B, -1))
            features = self.late_fusion_fn(features, features_rgb)

        if (obs_embed := tensordict["obs"].get("embed")) is not None:
            # concatenate along N dimension of embedding
            # obs_embed: (B, N, D)
            features = cat_nested([obs_embed, features], dim=-2)

        tensordict["obs", "embed"] = features
        
        return tensordict