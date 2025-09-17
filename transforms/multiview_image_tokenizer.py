from __future__ import annotations

import logging
from typing import Callable, Literal

import torch
import torch.nn as nn
from tensordict import TensorDict

from environments.specs import CameraSpec, DataSpecs, DepthStream, EmbedSpec, RGBStream
from transforms.base_transform import Transform
from utils.nested import cat_nested

log = logging.getLogger(__name__)


class MultiviewImageTokenizer(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        image_encoder: Callable[[int, int], nn.Module],
        embed_dim: int,
        image_type: Literal["rgb", "depth", "rgbd"] = "rgb",
        shared_encoder: bool = False,
        spatial_encoder: nn.Linear | None = None,
    ):
        super().__init__()

        if image_type == "rgb":
            stream_types = (RGBStream,)
            in_channels = 3
            if spatial_encoder:
                raise NotImplementedError(
                    "spatial_encoder is not implemented for rgb images"
                )
        elif image_type == "depth":
            stream_types = (DepthStream,)
            in_channels = 1 if spatial_encoder is None else spatial_encoder.out_features
        elif image_type == "rgbd":
            stream_types = (RGBStream, DepthStream)
            in_channels = (
                4 if spatial_encoder is None else 3 + spatial_encoder.out_features
            )
        else:
            raise ValueError(
                f"image_type must be one of 'rgb', 'depth', or 'rgbd', but got {image_type}"
            )
        self.image_type = image_type
        self.stream_types = stream_types
        self.spatial_encoder = spatial_encoder

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

        # Check if camera specs contain multiple rgb streams and print out a warning
        if RGBStream in stream_types:
            for key, spec in input_specs.items():
                rgb_streams = [
                    name
                    for name, stream in spec.streams.items()
                    if isinstance(stream, RGBStream)
                ]
                if len(rgb_streams) > 1:
                    log.warning(
                        f"Camera spec '{key}' contains multiple RGB streams. "
                        f"Only the first RGB stream {rgb_streams[0]} will be used"
                    )

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

        # instantiate rgb model(s)
        if shared_encoder:
            # verify that all streams have the same resolution
            if not all(
                stream.height_width == input_streams[0].height_width
                for stream in input_streams
            ):
                raise ValueError(
                    "All input streams must have the same height and width when using a shared encoder."
                    f"Got {[stream.height_width for stream in input_streams]}"
                )

            self.model = image_encoder(in_channels, embed_dim)
        else:
            self.models = nn.ModuleDict()
            for key in input_specs.keys():
                self.models[key] = image_encoder(in_channels, embed_dim)
        self.shared_encoder = shared_encoder

        # each camera produces one token
        # if we have stereo rgb, we just take the left camera
        new_spec = EmbedSpec(embed_dim=embed_dim, n_tokens=len(input_specs))

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

        imgs = []
        for key, spec in self._input_specs.items():
            # TODO: loop across stream types and find the first matching stream for each
            streams = []
            found = {stream_type: False for stream_type in self.stream_types}
            for name, stream in spec.streams.items():
                if not isinstance(stream, self.stream_types) or found[type(stream)]:
                    continue

                image = tensordict["obs", key, name]

                if stream.channel_order == "HWC":
                    image = torch.movedim(image, -1, -3)
                elif stream.channel_order == "HW":
                    if self.spatial_encoder and isinstance(stream, DepthStream):
                        # apply fourier-feature encoding to depth map
                        image = torch.unsqueeze(image, dim=-1)
                        assert image.shape[-1] == self.spatial_encoder.in_features, (
                            f"Expected depth image last dimension to be {self.spatial_encoder.in_features}, "
                            f"but got {image.shape[-1]} for {key}/{name}"
                        )
                        image = self.spatial_encoder(image)
                        image = torch.movedim(image, -1, -3)
                    else:
                        image = torch.unsqueeze(image, dim=-3)

                assert (
                    image.shape[-2:] == stream.height_width
                ), f"Expected image shape {stream.height_width}, but got {image.shape[-2:]} for {key}/{name}"

                if image.dtype == torch.uint8:
                    image = image.to(dtype=default_float_dtype).div(255)

                found[type(stream)] = True

                streams.append(image)

            if len(streams) > 1:
                # stack rgb and depth in channel dimension
                imgs.append(torch.cat(streams, dim=-3))
                assert len(streams) <= 2
            else:
                imgs.append(streams[0])

        if self.shared_encoder:
            # pass all rgb obs to rgb model

            # we stack and flatten rather than concatenate, to keep images from the same time step together
            # (B, T, C, H, W) -> (B, T, N, C, H, W)
            img = torch.stack(imgs, dim=-4)

            B, img_shape = (img.shape[0], img.shape[-3:])
            # (B, T, N, C, H, W) -> (B*T*N, C, H, W)
            img = img.view(-1, *img_shape)

            # (B*T*N, C, H, W) -> (B*T*N, D)
            features = self.model(img)
            # keep time and number of cameras flattened
            # (B*T*N, D) -> (B, T*N, D)
            features = torch.unflatten(features, dim=0, sizes=(B, -1))

        else:
            # run each rgb obs to independent models
            img = imgs[0]
            B, img_shape = (img.shape[0], img.shape[-3:])

            features = []
            for img, key in zip(imgs, self._input_specs.keys()):
                # (B, T, C, H, W) -> (B*T, C, H, W)
                img = img.view(-1, *img_shape)

                # (B*T, C, H, W) -> (B*T, D)
                feature = self.models[key](img)
                features.append(feature)

            # (B*T, D) -> (B*T, N, D)
            features = torch.stack(features, dim=1)
            # (B*T, N, D) -> (B*T*N, D)
            features = torch.flatten(features, end_dim=1)
            # (B*T*N, D) -> (B, T*N, D)
            features = torch.unflatten(features, dim=0, sizes=(B, -1))

        if (obs_embed := tensordict["obs"].get("embed")) is not None:
            # concatenate along N dimension of embedding
            # obs_embed: (B, N, D)
            features = cat_nested([obs_embed, features], dim=-2)

        tensordict["obs", "embed"] = features

        return tensordict
