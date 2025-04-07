from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

from environments.specs import DataSpecs, RGBCameraSpec, Spec
from transforms.base_transform import KeyMapping, Transform

if TYPE_CHECKING:
    from torch import Tensor
    from torch.nn import Module

    from environments.specs import DataSpecs


class MultiRgbTokenizer(Transform, nn.Module):
    def __init__(
        self,
        specs: DataSpecs,
        rgb_model: Callable[[], Module],
        embed_dim: int,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool = False,
    ):
        super().__init__()

        input_specs = {
            key: spec
            for key, spec in specs.obs.items()
            if isinstance(spec, RGBCameraSpec)
        }

        # instantiate rgb model(s)
        if share_rgb_model:
            first_key, spec = next(iter(input_specs.items()))
            im_shape = spec.shape
            for key, spec in input_specs.items():
                if spec.shape != im_shape:
                    raise ValueError(
                        f"Input specs {first_key} and {key} have different image shapes."
                    )

            self.model = rgb_model()
        else:
            self.models = nn.ModuleDict()
            for key in input_specs:
                self.models[key] = rgb_model()

            # store the rgb_keys so we can use them in the call method
            self._rgb_keys = list(input_specs.keys())

        self.share_rgb_model = share_rgb_model

        # create a modified specs object for the output
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        # the leading dim is the number of observed time steps
        # each camera produces one token
        # if we have stereo rgb, we just take the left camera
        embed_seq_len = sum(spec.shape[0] for spec in input_specs.values())
        obs_specs["embed"] = Spec(shape=(embed_seq_len, embed_dim), type="embed")
        self._output_specs = specs.replace(obs=obs_specs)

        # create a list of key mappings for the forward call
        nested_keys = []
        for key, spec in input_specs.items():
            # if we get a stereo camera, we just take the left camera
            subkey = spec.rgb_subkeys[0]
            if subkey is not None:
                nested_keys.append(("obs", key, subkey))
            else:
                nested_keys.append(("obs", key))
        self._key_mappings = [
            KeyMapping(
                in_keys=list(nested_keys),
                out_keys=[("obs", "embed")],
            )
        ]

    @property
    def key_mappings(self) -> list[KeyMapping]:
        return self._key_mappings

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def _call_one(self, *imgs: Tensor) -> Tensor:
        if self.share_rgb_model:
            # pass all rgb obs to rgb model

            # we stack and flatten rather than concatenate, to keep images from the same time step together
            # [B,T,C,H,W] -> [B,T,N,C,H,W]
            img = torch.stack(imgs, dim=-4)

            leading_dims, N, img_shape = (img.shape[:-4], img.shape[-4], img.shape[-3:])
            # [B,T,N,C,H,W] -> [B*T*N,C,H,W]
            img = img.view(-1, *img_shape)

            # [B*T*N,C,H,W] -> [B*T*N,D]
            features = self.model(img)
            # [B*T*N,D] -> [B,T,N,D]
            features = features.view(*leading_dims, N, -1)
            return features

        else:
            # run each rgb obs to independent models
            img = imgs[0]
            leading_dims, img_shape = img.shape[:-3], img.shape[-3:]
            features = []
            for img, key in zip(imgs, self._rgb_keys):
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
