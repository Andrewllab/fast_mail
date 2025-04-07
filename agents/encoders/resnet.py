from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, TypeVar

import torch.nn as nn
import torchvision

if TYPE_CHECKING:
    from torchvision.models import ResNet


log = logging.getLogger(__name__)


def beso_resnet_encoder(
    embed_dim: int,
    pretrained_weights: str | None = None,
    freeze_backbone: bool = False,
    use_group_norm: bool = True,
    **resnet_kwargs,
) -> ResNet:

    if pretrained_weights is None:
        # if we are loading pretrained weights, layer sizes must match the pretrained model
        # so we only set this if we are not using pretrained weights
        resnet_kwargs["num_classes"] = embed_dim

    model = torchvision.models.resnet18(weights=pretrained_weights, **resnet_kwargs)

    if use_group_norm:

        def make_group_norm(bn: nn.Module) -> nn.GroupNorm:
            assert isinstance(bn, nn.BatchNorm2d)
            gn = nn.GroupNorm(
                num_groups=bn.num_features // 16, num_channels=bn.num_features
            )

            # copy the parameters from the batch norm layer
            gn.weight.data = bn.weight.data.clone()
            gn.bias.data = bn.bias.data.clone()
            return gn

        model = replace(
            model,
            filter=lambda m: isinstance(m, nn.BatchNorm2d),
            fn=make_group_norm,
        )

    if pretrained_weights is not None:
        # have to replace the output layer if it has the wrong dimensionality
        # (e.g. if pretrained weights were loaded)
        model.fc = nn.Linear(model.fc.in_features, embed_dim)

    if freeze_backbone:
        # freeze all layers except the last one
        model.requires_grad_(False)
        model.fc.requires_grad_(True)

    return model


T = TypeVar("T", bound=nn.Module)


def replace(
    module: T,
    filter: Callable[[nn.Module], bool],
    fn: Callable[[nn.Module], nn.Module],
) -> T:
    """Recursively replace modules in a PyTorch module.

    Args:
        module (nn.Module): The module to modify.
        filter (Callable[[nn.Module], bool]): A function that returns True for modules to be replaced.
        fn (Callable[[nn.Module], nn.Module]): A function that takes a module and returns a modified module.

    Returns:
        nn.Module: The modified module.
    """
    for name, child in module.named_children():
        if filter(child):
            setattr(module, name, fn(child))
        else:
            replace(child, filter, fn)
    return module
