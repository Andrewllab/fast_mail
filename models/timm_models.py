from __future__ import annotations

from timm import create_model


def create_timm_model(
    input_channels: int,
    embed_dim: int,
    model_name: str,
    pretrained: bool = False,
    **convnext_kwargs,
):
    model = create_model(
        model_name,
        pretrained=pretrained,
        in_chans=input_channels,
        num_classes=embed_dim,
        **convnext_kwargs,
    )

    return model
