from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

from environments.specs import (
    DataSpecs,
    RGBCameraSpec,
    RGBDCameraSpec,
    Spec,
    StereoRGBCameraSpec,
)
from transforms.base_transform import KeyMapping, Transform

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

    from environments.specs import DataSpecs


class MultiRgbTokenizer(nn.Module, Transform):
    def __init__(
        self,
        specs: DataSpecs,
        rgb_model: Callable[[], Module],
        embed_dim: int,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = False,
    ):
        super().__init__()

        self._input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, RGBCameraSpec)
        }

        # handle sharing vision backbone
        if share_rgb_model:
            self.model = rgb_model()
        else:
            self.models = nn.ModuleDict()
            for key in self._input_specs:
                self.models[key] = rgb_model()

        self.share_rgb_model = share_rgb_model

        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        # the leading dim is the number of observed time steps
        # each camera produces one token
        embed_seq_len = sum(spec.shape[0] for spec in self._input_specs.values())
        obs_specs["embed"] = Spec(shape=(embed_seq_len, embed_dim), type="embed")
        self._specs = specs.replace(obs=obs_specs)

        nested_keys = []
        for key, spec in self._input_specs.items():
            if isinstance(spec, RGBDCameraSpec):
                nested_keys.append(("obs", key, "rgb"))
            elif isinstance(spec, StereoRGBCameraSpec):
                nested_keys.extend([("obs", key, "left"), ("obs", key, "right")])
            else:
                nested_keys.append(("obs", key))

        self._nested_keys = nested_keys

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return [
            KeyMapping(
                in_keys=list(self._nested_keys),
                out_keys=[("obs", "embed")],
            )
        ]

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def forward(self, *imgs: Tensor) -> Tensor:
        if self.share_rgb_model:
            # pass all rgb obs to rgb model

            # we stack and flatten rather than concatenate, to keep images from the same time step together
            # [B,T,C,H,W] -> [B,T,N,C,H,W]
            img = torch.stack(imgs, dim=-4)

            leading_dims, N, img_shape = (img.shape[:-4], img.shape[-4], img.shape[-3:])
            # [B,T,N,C,H,W] -> [B*T*N,C,H,W]
            img = img.view(-1, *img_shape)

            # [B*T*N,C,H,W] -> [B*T*N,D]
            features = self.model(imgs)
            # [B*T*N,D] -> [B,T,N,D]
            features = features.view(*leading_dims, -1)
            return features

        else:
            # run each rgb obs to independent models
            features = []
            for img, (key, spec) in zip(imgs, self._input_specs.items()):
                assert tuple(img.shape[1:]) == spec.shape
                leading_dims, img_shape = img.shape[:-3], img.shape[-3:]
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

            return features
