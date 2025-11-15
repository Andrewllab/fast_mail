from __future__ import annotations

import logging
from typing import Callable, Literal, Sequence

import torch
import torch.nn as nn
from tensordict import TensorDict

from environments.specs import (
    CameraSpec,
    DataSpecs,
    DepthStream,
    EmbedSpec,
    PointMapStream,
    RGBStream,
)
from transforms.base_transform import Transform
from utils.nested import cat_nested

IMAGE_TYPES = Literal["rgb", "depth", "pointmap"]

log = logging.getLogger(__name__)


class ConvolutionalTokenizer(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        embed_dim: int,
        image_type: IMAGE_TYPES | Sequence[IMAGE_TYPES],
        encoder: Callable[[int, int], nn.Module] | None = None,
        depth_encoder: Callable[[int, int], nn.Module] | None = None,
        pointmap_encoder: Callable[[int, int], nn.Module] | None = None,
        fusion_type: Literal["channel", "add", "cat"] = "cat",
        shared_encoder: bool = False,
        spatial_encoder: Callable[[int], nn.Linear] | None = None,
        token_pos_encoder: Callable[[int, int], nn.Module] | None = None,
    ):
        super().__init__()

        # determine required stream types
        if isinstance(image_type, str):
            image_type = [image_type]
        stream_types = []
        if "rgb" in image_type:
            stream_types.append(RGBStream)
        if "depth" in image_type:
            stream_types.append(DepthStream)
        if "pointmap" in image_type:
            stream_types.append(PointMapStream)

        if not stream_types:
            raise ValueError(
                "At least one of 'rgb', 'depth', or 'pointmap' must be specified."
            )

        # find all camera specs that contain at least one of each required
        # stream type
        input_specs = {}
        for key, spec in specs.obs.items():
            if not isinstance(spec, CameraSpec):
                continue

            cam_streams = {}

            for stream_type in stream_types:
                streams = [
                    (name, stream)
                    for name, stream in spec.streams.items()
                    if isinstance(stream, stream_type)
                ]

                if not streams:
                    # at least one stream type is missing, skip this camera
                    break

                if len(streams) > 1:
                    log.warning(
                        f"Camera spec '{key}' contains multiple {stream_type.__name__}s. "
                        f"Only the first {stream_type.__name__} '{streams[0][0]}' will be used"
                    )

                name, stream = streams[0]
                cam_streams[name] = (spec, stream)

            else:
                # at least one stream was found for every required type
                input_specs[key] = cam_streams

        if not input_specs:
            raise ValueError("No valid input specs found.")

        # verify that all streams have the same time dimension
        times = [
            stream.time
            for streams in input_specs.values()
            for (spec, stream) in streams.values()
        ]
        if not all(time == times[0] for time in times):
            raise ValueError(
                f"All input streams must have the same time dimension. Got {times}"
            )

        # verify that all point map streams have 3 channels
        pointmap_in_channels = [
            stream.channels
            for streams in input_specs.values()
            for (spec, stream) in streams.values()
            if isinstance(stream, PointMapStream)
        ]
        if not all(in_ch == 3 for in_ch in pointmap_in_channels):
            raise ValueError(
                f"All input PointMap streams must have 3 channels. Got {pointmap_in_channels}"
            )

        # determine input channels for each stream type and instantiate spatial
        # encoders if needed
        in_channels = {}
        spatial_encoders = nn.ModuleDict()
        if "rgb" in image_type:
            in_channels["rgb"] = 3
        if "depth" in image_type:
            channels = 1
            if spatial_encoder is not None:
                spatial_encoders["depth"] = spatial_encoder(channels)
                channels = spatial_encoders["depth"].out_features
            in_channels["depth"] = channels
        if "pointmap" in image_type:
            channels = 3
            if spatial_encoder is not None:
                spatial_encoders["pointmap"] = spatial_encoder(channels)
                channels = spatial_encoders["pointmap"].out_features
            in_channels["pointmap"] = channels

        # instantiate convolutional encoder(s)
        if fusion_type == "channel":
            if encoder is None:
                raise ValueError(
                    "`encoder` must be provided for channel fusion of multiple stream types"
                )

            total_in_channels = sum(in_channels.values())

            if shared_encoder:
                height_widths = [
                    stream.height_width
                    for streams in input_specs.values()
                    for (spec, stream) in streams.values()
                ]
                if not all(hw == height_widths[0] for hw in height_widths):
                    raise ValueError(
                        f"All input streams must have the same height and width when using channel fusion and a shared encoder. Got {height_widths}"
                    )

                conv_encoder = encoder(total_in_channels, embed_dim)
            else:
                for key, cam_streams in input_specs.items():
                    height_widths = [
                        stream.height_width for (spec, stream) in cam_streams.values()
                    ]
                    if not all(hw == height_widths[0] for hw in height_widths):
                        raise ValueError(
                            f"Input streams from {key} must have the same height and width when using channel fusion. Got {height_widths}"
                        )

                conv_encoder = nn.ModuleDict(
                    {
                        key: encoder(total_in_channels, embed_dim)
                        for key in input_specs.keys()
                    }
                )

        elif fusion_type in ["add", "cat"]:
            conv_encoder = nn.ModuleDict()

            if shared_encoder:
                for t in stream_types:
                    height_widths = [
                        stream.height_width
                        for streams in input_specs.values()
                        for (spec, stream) in streams.values()
                        if isinstance(stream, t)
                    ]
                    if not all(hw == height_widths[0] for hw in height_widths):
                        raise ValueError(
                            f"All input {stream_type.__name__}s must have the same height and width when using a shared encoder. Got {height_widths}"
                        )

                build = lambda cls, in_ch: cls(in_ch, embed_dim)

            else:
                build = lambda cls, in_ch: nn.ModuleDict(
                    {key: cls(in_ch, embed_dim) for key in input_specs.keys()}
                )

            if "rgb" in image_type:
                if encoder is None:
                    raise ValueError(
                        "`encoder` must be provided for tokenizing rgb streams"
                    )
                # __name__ = "RGBStream", since the key must be a string for ModuleDict
                conv_encoder[RGBStream.__name__] = build(encoder, in_channels["rgb"])

            if "depth" in image_type:
                depth_encoder = depth_encoder or encoder
                if depth_encoder is None:
                    raise ValueError(
                        "`depth_encoder` or `encoder` must be provided for tokenizing depth streams"
                    )
                conv_encoder[DepthStream.__name__] = build(
                    depth_encoder, in_channels["depth"]
                )

            if "pointmap" in image_type:
                pointmap_encoder = pointmap_encoder or encoder
                if pointmap_encoder is None:
                    raise ValueError(
                        "`pointmap_encoder` or `encoder` must be provided for tokenizing pointmap streams"
                    )
                conv_encoder[PointMapStream.__name__] = build(
                    pointmap_encoder, in_channels["pointmap"]
                )
        else:
            raise ValueError(
                f"fusion_type must be one of 'channel', 'add', or 'cat', but got {fusion_type}"
            )

        self.stream_types = stream_types
        self.fusion_type = fusion_type
        self.shared_encoder = shared_encoder
        self.spatial_encoders = spatial_encoders
        self.conv_encoder = conv_encoder
        self._input_specs = input_specs

        # determine number of output tokens
        # each camera produces:
        #   "channel" or "cat" fusion: one token per modality
        #   "add" fusion: one token
        n_tokens = len(input_specs) * (len(stream_types) if fusion_type == "cat" else 1)

        # create encoder for token position
        if token_pos_encoder is None:
            log.warning("No token position encoder provided. Using nn.Embedding.")
            token_pos_encoder = nn.Embedding
        self.token_pos_encoder = token_pos_encoder(n_tokens, embed_dim)

        # create a modified specs object for the output
        new_spec = EmbedSpec(embed_dim=embed_dim, n_tokens=n_tokens)
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        if "embed" in obs_specs:
            embed_spec = obs_specs["embed"]
            assert isinstance(embed_spec, EmbedSpec)
            new_spec = embed_spec.concat(new_spec)
        obs_specs["embed"] = new_spec
        self._output_specs = specs.replace(obs=obs_specs)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def forward(self, tensordict: TensorDict) -> TensorDict:
        default_float_dtype = torch.get_default_dtype()

        imgs = {}
        for key, streams in self._input_specs.items():

            cam_imgs = {}
            for name, (spec, stream) in streams.items():
                image = tensordict["obs", key, name]

                if isinstance(stream, PointMapStream):
                    if "pointmap" in self.spatial_encoders:
                        encoder = self.spatial_encoders["pointmap"]
                        assert image.shape[-1] == encoder.in_features == 3
                        image = encoder(image)
                    image = torch.movedim(image, -1, -3)

                elif isinstance(stream, DepthStream):
                    if "depth" in self.spatial_encoders:
                        image = image.unsqueeze(dim=-1)
                        encoder = self.spatial_encoders["depth"]
                        assert image.shape[-1] == encoder.in_features == 1
                        image = encoder(image)
                        image = torch.movedim(image, -1, -3)
                    else:
                        image = torch.unsqueeze(image, dim=-3)

                # RGBStream or IRStream
                elif stream.channel_order == "HWC":
                    image = torch.movedim(image, -1, -3)
                elif stream.channel_order == "HW":
                    image = torch.unsqueeze(image, dim=-3)

                assert (
                    image.shape[-2:] == stream.height_width
                ), f"Expected image shape {stream.height_width}, but got {image.shape[-2:]} for {key}/{name}"

                if image.dtype == torch.uint8:
                    image = image.to(dtype=default_float_dtype).div(255)

                assert type(stream) in self.stream_types
                cam_imgs[type(stream)] = image

            imgs[key] = cam_imgs

        # img: (B, T, C, H, W)

        if self.fusion_type == "channel":
            # concatenate images in channel dimension
            imgs = {
                key: cat(list(cam_imgs.values()), dim=-3)
                for key, cam_imgs in imgs.items()
            }

            if self.shared_encoder:
                # stack images from all cameras for each time step
                # (B, T, C, H, W) -> (B, T, N, C, H, W)
                imgs = stack(list(imgs.values()), dim=-4)

                leading_dims, shape = (imgs.shape[:-3], imgs.shape[-3:])
                # (B, T, N, C, H, W) -> (B*T*N, C, H, W)
                imgs = imgs.view(-1, *shape)
                # (B*T*N, C, H, W) -> (B*T*N, D)
                features = self.conv_encoder(imgs)
                # (B*T*N, D) -> (B, T, N, D)
                features = features.unflatten(dim=0, sizes=leading_dims)

            else:
                all_features = []

                for key, cam_img in imgs.items():
                    leading_dims, shape = (cam_img.shape[:-3], cam_img.shape[-3:])

                    # (B, T, C, H, W) -> (B*T, C, H, W)
                    cam_img = cam_img.view(-1, *shape)
                    # (B*T, C, H, W) -> (B*T, D)
                    features = self.conv_encoder[key](cam_img)
                    # (B*T, D) -> (B, T, D)
                    features = features.unflatten(dim=0, sizes=leading_dims)
                    all_features.append(features)

                # (B, T, D) -> (B, T, N, D)
                features = stack(all_features, dim=-2)

        else:
            # group images by stream type instead of camera
            imgs = {
                t: {key: cam_imgs[t] for key, cam_imgs in imgs.items()}
                for t in self.stream_types
            }
            all_features = []

            for t, t_imgs in imgs.items():
                encoder = self.conv_encoder[t.__name__]

                if self.shared_encoder:
                    # (B, T, C, H, W) -> (B, T, N, C, H, W)
                    t_imgs = stack(list(t_imgs.values()), dim=-4)

                    leading_dims, shape = (t_imgs.shape[:-3], t_imgs.shape[-3:])
                    # (B, T, N, C, H, W) -> (B*T*N, C, H, W)
                    t_imgs = t_imgs.view(-1, *shape)
                    # (B*T*N, C, H, W) -> (B*T*N, D)
                    t_features = encoder(t_imgs)
                    # (B*T*N, D) -> (B, T, N, D)
                    t_features = t_features.unflatten(dim=0, sizes=leading_dims)

                    all_features.append(t_features)

                else:
                    t_features = []

                    for key, img in t_imgs.items():
                        leading_dims, shape = (img.shape[:-3], img.shape[-3:])

                        # (B, T, C, H, W) -> (B*T, C, H, W)
                        img = img.view(-1, *shape)
                        # (B*T, C, H, W) -> (B*T, D)
                        feat = encoder[key](img)
                        # (B*T, D) -> (B, T, D)
                        feat = feat.unflatten(dim=0, sizes=leading_dims)
                        t_features.append(feat)

                    # (B, T, D) -> (B, T, N, D)
                    t_features = stack(t_features, dim=-2)
                    all_features.append(t_features)

            # features: (B, T, N, D)

            if len(all_features) == 1:
                features = all_features[0]
            elif self.fusion_type == "cat":
                # concatenate features from each modality along token dimension
                features = torch.cat(all_features, dim=-2)
            elif self.fusion_type == "add":
                # sum features from each modality into one token per camera
                features = torch.stack(all_features).sum(dim=0)

        # flatten time and token dimensions
        # (B, T, N, D) -> (B, T*N, D)
        features = features.flatten(start_dim=1, end_dim=2)

        # add encoding of the token position to each token
        N = features.shape[1]
        token_indices = torch.arange(N, dtype=torch.long, device=features.device)
        token_pos_embed = self.token_pos_encoder(token_indices)
        features += token_pos_embed

        if (obs_embed := tensordict["obs"].get("embed")) is not None:
            # concatenate along N dimension of embedding
            # obs_embed: (B, N, D)
            features = cat_nested([obs_embed, features], dim=-2)

        tensordict["obs", "embed"] = features

        return tensordict


def stack(tensors: Sequence[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Stacks a sequence of tensors along a new dimension.

    Args:
        tensors (Sequence[torch.Tensor]): Sequence of tensors to stack.
        dim (int, optional): Dimension along which to stack the tensors. Defaults to 0.

    Returns:
        torch.Tensor: Stacked tensor.
    """
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim=dim)
    return torch.stack(tensors, dim=dim)


def cat(tensors: Sequence[torch.Tensor], dim: int = 0) -> torch.Tensor:
    """Concatenates a sequence of tensors along an existing dimension.

    Args:
        tensors (Sequence[torch.Tensor]): Sequence of tensors to concatenate.
        dim (int, optional): Dimension along which to concatenate the tensors. Defaults to 0.

    Returns:
        torch.Tensor: Concatenated tensor.
    """
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim=dim)
