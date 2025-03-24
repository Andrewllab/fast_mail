from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
import torchvision

if TYPE_CHECKING:
    from torchvision.models import ResNet


def beso_resnet_encoder(
    embed_dim: int,
    pretrained_weights: str | None,
    freeze_backbone: bool | None = None,
    **resnet_kwargs,
) -> ResNet:

    # by default, only freeze weights if we load a pretrained model
    if freeze_backbone is None:
        freeze_backbone = pretrained_weights is not None

    # ResNet supports other norm layers besides the default BatchNorm2d, but
    # instantiates them with a single argument `num_features`. Therefore we
    # have to adapt the constructor for GroupNorm.
    def make_group_norm(num_features: int):
        return nn.GroupNorm(num_groups=num_features // 16, num_channels=num_features)

    if pretrained_weights is None:
        # if we are loading pretrained weights, layer sizes must match the pretrained model
        # so we only set this if we are not using pretrained weights
        resnet_kwargs["num_classes"] = embed_dim

    model = torchvision.models.resnet18(
        weights=pretrained_weights, norm_layer=make_group_norm, **resnet_kwargs
    )

    if freeze_backbone:
        model.requires_grad_(False)

    if model.fc.out_features != embed_dim:
        # have to replace the output layer if it has the wrong dimensionality
        # (e.g. if pretrained weights were loaded)
        model.fc = nn.Linear(model.fc.in_features, model.fc.out_features)
    elif freeze_backbone:
        # output layer has the right dimensionality, but we have to unfreeze it
        model.fc.requires_grad_(True)

    return model
