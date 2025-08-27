from __future__ import annotations

from functools import partial

import torch.nn as nn
import torchvision
from torchvision.models.convnext import ConvNeXt, LayerNorm2d
from torchvision.ops.misc import Conv2dNormActivation


def convnext_encoder(
    input_channels: int,
    embed_dim: int,
    pretrained_weights: str | None = None,
    freeze_backbone: bool = False,
    **convnext_kwargs,
) -> ConvNeXt:
    """Create a ConvNeXt encoder with specified parameters."""
    if pretrained_weights is None:
        convnext_kwargs["num_classes"] = embed_dim
    else:
        assert (
            input_channels == 3
        ), "Pretrained weights are only available for 3 input channels."

    model = torchvision.models.convnext_tiny(
        weights=pretrained_weights, **convnext_kwargs
    )

    if input_channels != 3:
        # replace the first conv layer with one that has the correct number of input channels
        out_channels = model.features[0][0].out_channels
        model.features[0][0] = Conv2dNormActivation(
            in_channels=input_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=4,
            padding=0,
            norm_layer=partial(LayerNorm2d, eps=1e-6),
            activation_layer=None,
            bias=True,
        )

    if pretrained_weights is not None:
        # have to replace the output layer if it has the wrong dimensionality
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, embed_dim)

    if freeze_backbone:
        # freeze all layers except the last one
        model.requires_grad_(False)
        model.classifier[2].requires_grad_(True)

    return model
