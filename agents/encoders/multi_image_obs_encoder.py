from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
import torchvision

from agents.encoders.crop_randomizer import CropRandomizer

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

    from environments.datasets.base_dataset import TrajectoryDataset


class MultiImageObsEncoder(nn.Module):
    def __init__(
        self,
        rgb_model: Callable[[], Module],
        embed_dim: int,
        dataset: TrajectoryDataset,
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

        self._embed_dim = embed_dim

        # handle sharing vision backbone
        if share_rgb_model:
            self.model = rgb_model()
        else:
            self.models = nn.ModuleDict()

        self.transforms = nn.ModuleDict()

        obs_space = dataset.obs_space
        self.rgb_obs_space = {
            key: info for key, info in obs_space.items() if info["type"] == "rgb"
        }

        for key, info in self.rgb_obs_space.items():
            if not share_rgb_model:
                self.models[key] = rgb_model()

            transform = []

            # configure resize
            input_shape = info["shape"]
            if resize_shape is not None:
                if isinstance(resize_shape, dict):
                    h, w = resize_shape[key]
                else:
                    h, w = resize_shape
                transform.append(torchvision.transforms.Resize(size=(h, w)))

                # update input_shape for next transform
                input_shape = (input_shape[0], h, w)

            # configure randomizer
            if crop_shape is not None:
                if isinstance(crop_shape, dict):
                    h, w = crop_shape[key]
                else:
                    h, w = crop_shape
                if random_crop:
                    randomizer = CropRandomizer(
                        input_shape=input_shape,
                        crop_height=h,
                        crop_width=w,
                        num_crops=1,
                        pos_enc=False,
                    )
                else:
                    randomizer = torchvision.transforms.CenterCrop(size=(h, w))
                transform.append(randomizer)

            # configure normalizer
            if imagenet_norm:
                transform.append(
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    )
                )

            self.transforms[key] = nn.Sequential(*transform)

        self.share_rgb_model = share_rgb_model

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def embed_seq_len(self) -> int:
        """Number of tokens in the output embedding."""
        # the leading dim is the number of observed time steps
        # each camera produces one token
        return sum(info.shape[0] for info in self.rgb_obs_space.values())

    def forward(self, obs: dict) -> Tensor:
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = []
            for key, info in self.rgb_obs_space:
                img = obs[key]

                leading_dims, img_shape = img[0].shape[:-3], img[0].shape[-3:]
                assert img_shape == info["shape"]
                # [B,T,C,H,W] -> [B*T,C,H,W]
                img = img.view(-1, *img_shape)

                img = self.transforms[key](img)
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
            for key, info in self.rgb_obs_space:
                img = obs[key]

                leading_dims, img_shape = img[0].shape[:-3], img[0].shape[-3:]
                assert img_shape == info["shape"]
                # [B,T,C,H,W] -> [B*T,C,H,W]
                img = img.view(-1, *img_shape)

                img = self.transforms[key](img)
                # [B*T,C,H,W] -> [B*T,D]
                feature = self.models[key](img)
                features.append(feature)

            # [B*T,D] -> [B*T,N,D]
            features = torch.stack(features, dim=1)
            # [B*T,N,D] -> [B,T,N,D]
            features = features.view(*leading_dims, -1)
            return features
