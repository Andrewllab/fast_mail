from __future__ import annotations

import torch.nn as nn
import torchvision
from torchvision.models.vision_transformer import VisionTransformer


def vit_encoder(
    embed_dim: int,
    input_channels: int = 3,
    pretrained_weights: str | None = None,
    freeze_backbone: bool = False,
    **vit_kwargs,
) -> VisionTransformer:
    """Create a Vision Transformer (ViT-B/16) encoder with specified parameters."""
    if pretrained_weights is None:
        vit_kwargs["num_classes"] = embed_dim
    else:
        assert (
            input_channels == 3
        ), "Pretrained weights are only available for 3 input channels."

    model = torchvision.models.vit_b_16(weights=pretrained_weights, **vit_kwargs)

    if input_channels != 3:
        old_conv_proj = model.conv_proj
        model.conv_proj = nn.Conv2d(
            in_channels=input_channels,
            out_channels=old_conv_proj.out_channels,
            kernel_size=old_conv_proj.kernel_size,
            stride=old_conv_proj.stride,
        )

    if pretrained_weights is not None:
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, embed_dim)

    if freeze_backbone:
        # Freeze all layers except the heads
        model.requires_grad_(False)
        model.heads.head.requires_grad_(True)

    return model
