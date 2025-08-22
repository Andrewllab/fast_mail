from __future__ import annotations

import torch.nn as nn
from timm import create_model

def davit_encoder(
    embed_dim: int,
    input_channels: int = 3,
    pretrained: bool = False,
    **davit_kwargs,
) -> nn.Module:
    if pretrained:
        assert input_channels == 3, "Pretrained weights are only available for 3 input channels."
    
    model = create_model(
        model_name='davit_tiny',
        pretrained=pretrained,
        in_chans=input_channels,
        num_classes=embed_dim,
        **davit_kwargs,
    )

    return model

