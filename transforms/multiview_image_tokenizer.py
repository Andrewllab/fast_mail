from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Literal

import torch
import torch.nn as nn
from torch import Tensor

from environments.specs import (
    DataSpecs,
    DepthCameraSpec,
    EmbedSpec,
    RGBCameraSpec,
    RGBDCameraSpec,
)
from transforms.base_transform import KeyMapping, Transform

log = logging.getLogger(__name__)


class MultiviewImageTokenizer(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        image_encoder: Callable[[], nn.Module],
        embed_dim: int,
        image_type: Literal["rgb", "depth", "rgbd"] = "rgb",
        shared_encoder: bool = False,
    ):
        super().__init__()

        if image_type == "rgb":
            allowed_type = RGBCameraSpec
        elif image_type == "depth":
            allowed_type = DepthCameraSpec
        elif image_type == "rgbd":
            allowed_type = RGBDCameraSpec
            raise NotImplementedError
        else:
            raise ValueError(
                f"image_type must be one of 'rgb', 'depth', or 'rgbd', but got {image_type}"
            )
        self.image_type = image_type

        input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, allowed_type)
        }

        # instantiate rgb model(s)
        if shared_encoder:
            first_key, spec = next(iter(input_specs.items()))
            im_shape = spec.shape
            for key, spec in input_specs.items():
                if spec.shape != im_shape:
                    raise ValueError(
                        f"Input specs {first_key} and {key} have different image shapes."
                    )

            self.model = image_encoder()
        else:
            self.models = nn.ModuleDict()
            for key in input_specs:
                self.models[key] = image_encoder()

            # store the rgb_keys so we can use them in the call method
            self._input_keys = list(input_specs.keys())

        self.shared_encoder = shared_encoder

        # the leading dim is the number of observed time steps
        Ts = [spec.shape[0] for spec in input_specs.values()]
        T = Ts[0]
        assert all(t == T for t in Ts)
        # each camera produces one token
        # if we have stereo rgb, we just take the left camera
        n_embed_tokens = len(input_specs)

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        if "embed" in obs_specs:
            # if embedding sequence has fixed length, increase length to account for new embedding tokens
            embed_spec = obs_specs["embed"]
            assert isinstance(embed_spec, EmbedSpec)
            assert len(embed_spec.shape) == 3
            assert embed_spec.shape[0] == T

            if embed_spec.fixed_shape:
                assert embed_spec.shape[1] is not None
                obs_specs["embed"] = dataclasses.replace(
                    embed_spec,
                    shape=(
                        T,
                        embed_spec.shape[1] + n_embed_tokens,
                        embed_spec.shape[2],
                    ),
                )
                log.debug(
                    f"Extended obs embedding spec to {n_embed_tokens} tokens per time step",
                )
        else:
            # create new embedding spec
            obs_specs["embed"] = EmbedSpec(shape=(T, n_embed_tokens, embed_dim))
            log.debug(
                f"Created obs embedding spec with {n_embed_tokens} tokens per time step",
            )
        self._output_specs = specs.replace(obs=obs_specs)

        # create a list of key mappings for the forward call
        nested_keys = []
        for key, spec in input_specs.items():
            # if we get a stereo camera, we just take the left camera
            if image_type == "rgb":
                assert isinstance(spec, RGBCameraSpec)
                subkey = spec.rgb_subkeys[0]
            elif image_type == "depth":
                assert isinstance(spec, DepthCameraSpec)
                subkey = spec.depth_subkeys[0]
            elif image_type == "rgbd":
                assert isinstance(spec, RGBDCameraSpec)
                raise NotImplementedError

            if subkey is not None:
                nested_keys.append(("obs", key, subkey))
            else:
                nested_keys.append(("obs", key))
        self._key_mappings = [
            KeyMapping(
                in_keys=[("obs", "embed")] + list(nested_keys),
                out_keys=[("obs", "embed")],
            )
        ]

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return self._key_mappings

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(self, obs_embed, *imgs: Tensor) -> Tensor:
        default_float_dtype = torch.get_default_dtype()

        if self.shared_encoder:
            # pass all rgb obs to rgb model

            # we stack and flatten rather than concatenate, to keep images from the same time step together
            # [B,T,C,H,W] -> [B,T,N,C,H,W]
            img = torch.stack(imgs, dim=-4)

            if img.shape[-1] == 3:
                img = torch.movedim(img, -1, -3)
            if img.dtype == torch.uint8:
                img = img.to(dtype=default_float_dtype).div(255)

            leading_dims, N, img_shape = (img.shape[:-4], img.shape[-4], img.shape[-3:])
            # [B,T,N,C,H,W] -> [B*T*N,C,H,W]
            img = img.view(-1, *img_shape)

            # [B*T*N,C,H,W] -> [B*T*N,D]
            features = self.model(img)
            # [B*T*N,D] -> [B,T,N,D]
            features = features.view(*leading_dims, N, -1)

        else:
            # run each rgb obs to independent models
            img = imgs[0]
            if self.image_type == "rgb":
                leading_dims, img_shape = img.shape[:-3], img.shape[-3:]
            elif self.image_type == "depth":
                leading_dims, img_shape = img.shape[:-2], img.shape[-2:]
                img_shape = (1,) + img_shape  # add singleton channel dim
            elif self.image_type == "rgbd":
                raise NotImplementedError

            if img.shape[-1] == 3:
                imgs = tuple(torch.movedim(im, -1, -3) for im in imgs)
            if img.dtype == torch.uint8:
                imgs = tuple(im.to(dtype=default_float_dtype).div(255) for im in imgs)

            features = []
            for img, key in zip(imgs, self._input_keys):
                # [B,T,C,H,W] -> [B*T,C,H,W]
                img = img.view(-1, *img_shape)

                # [B*T,C,H,W] -> [B*T,D]
                feature = self.models[key](img)
                features.append(feature)

            N = len(features)
            # [B*T,D] -> [B*T,N,D]
            features = torch.stack(features, dim=1)
            # [B*T,N,D] -> [B,T,N,D]
            features = features.view(*leading_dims, N, -1)

        if obs_embed is not None:
            # concatenate along N dimension of embedding, keeping tokens from the same time step together
            # obs_embed: (B, T, N, D)
            obs_embed = torch.cat([obs_embed, features], dim=2)
            return obs_embed
        else:
            return features
