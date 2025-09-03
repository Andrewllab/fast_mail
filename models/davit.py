from __future__ import annotations

from typing import Literal

import torch.nn as nn
from timm import create_model


def davit_encoder(
    input_channels: int,
    embed_dim: int,
    pretrained_weights: Literal["IMAGENET1K_V1"] | None = None,
    **davit_kwargs,
) -> nn.Module:
    if pretrained_weights is not None:
        assert (
            input_channels == 3
        ), "Pretrained weights are only available for 3 input channels."

    model = create_model(
        model_name="davit_tiny",
        pretrained=pretrained_weights == "IMAGENET1K_V1",
        in_chans=input_channels,
        num_classes=embed_dim,
        **davit_kwargs,
    )

    return model
