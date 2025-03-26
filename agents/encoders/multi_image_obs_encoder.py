from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from torch.nn import Module

from environments.datasets.base_dataset import TrajectoryDataset
from environments.specs import DataSpecs, Spec
from transforms.crop_randomizer import CropRandomizer


class MultiImageObsEncoder(nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        rgb_model: Callable[[], Module],
        embed_dim: int,
        resize_shape: tuple[int, int] | dict[str, tuple] | None = None,
        crop_shape: tuple[int, int] | dict[str, tuple] | None = None,
        random_crop: bool = True,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = False,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm: bool = False,
    ):
        super().__init__()

        self.rgb_specs = {
            key: spec for key, spec in specs.obs.items() if spec.type == "rgb"
        }

        # handle sharing vision backbone
        if share_rgb_model:
            self.model = rgb_model()
        else:
            self.models = nn.ModuleDict()
            for key in self.rgb_specs:
                self.models[key] = rgb_model()

        self.share_rgb_model = share_rgb_model

        self._obs_spec = dict(specs.obs)  # copy obs spec for local modification
        # the leading dim is the number of observed time steps
        # each camera produces one token
        embed_seq_len = sum(spec.shape[0] for spec in self.rgb_specs.values())
        self._obs_spec["obs_embed"] = Spec(
            shape=(embed_seq_len, embed_dim), type="embed"
        )
        self._specs = DataSpecs(_obs=self._obs_spec, action=specs.action)

    @property
    def specs(self) -> DataSpecs:
        return self._specs

    def forward(self, obs: dict) -> Tensor:
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = []
            for key, spec in self.rgb_specs.items():
                img = obs[key]

                assert tuple(img.shape[1:]) == spec.shape
                leading_dims, img_shape = img.shape[:-3], img.shape[-3:]
                # [B,T,C,H,W] -> [B*T,C,H,W]
                img = img.view(-1, *img_shape)

                imgs.append(img)

            # we stack and flatten rather than concatenate, to keep images from the same time step together
            # [B*T,C,H,W] -> [B*T,N,C,H,W]
            imgs = torch.stack(imgs, dim=1)
            # [B*T,N,C,H,W] -> [B*T*N,C,H,W]
            imgs = imgs.view(-1, *img_shape)

            # [B*T*N,C,H,W] -> [B*T*N,D]
            features = self.model(imgs)
            # [B*T*N,D] -> [B,T,N,D]
            features = features.view(*leading_dims, -1)
            return features

        else:
            # run each rgb obs to independent models
            features = []
            for key, spec in self.rgb_specs.items():
                img = obs[key]

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
