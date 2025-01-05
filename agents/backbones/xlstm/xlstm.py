import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import einops
import math
from typing import Optional
from torch.nn import functional as F
import logging
from dacite import from_dict
from dacite import Config as DaciteConfig
from xlstm import xLSTMLMModel, xLSTMLMModelConfig


logger = logging.getLogger(__name__)

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class xlstmEncoder(nn.Module):
    def __init__(
            self,
            xlstm_config: DictConfig,
            block_size: int,
            embed_dim: int,
            bias: bool = False,
                ):
        super().__init__()
        
        xlstm_config.context_length = block_size
        # Convert DictConfig to the required configuration objects
        print(f"xlstm_config:  {xlstm_config}")

        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=xlstm_config.mlstm_block.mlstm.conv1d_kernel_size,
                qkv_proj_blocksize=xlstm_config.mlstm_block.mlstm.qkv_proj_blocksize,
                num_heads=xlstm_config.mlstm_block.mlstm.num_heads,
                proj_factor=xlstm_config.mlstm_block.mlstm.proj_factor,
                dropout=xlstm_config.mlstm_block.mlstm.dropout
            )
        )
        if xlstm_config.slstm_block.enabled:
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=xlstm_config.slstm_block.slstm.backend,
                    num_heads=xlstm_config.slstm_block.slstm.num_heads,
                    conv1d_kernel_size=xlstm_config.slstm_block.slstm.conv1d_kernel_size,
                    bias_init=xlstm_config.slstm_block.slstm.bias_init,
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=xlstm_config.slstm_block.feedforward.proj_factor,
                    act_fn=xlstm_config.slstm_block.feedforward.act_fn
                ),
            )
            slstm_at = xlstm_config.slstm_at
        else:
            slstm_block = None
            slstm_at = []

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mlstm_block,
            slstm_block=slstm_block,
            context_length=xlstm_config.context_length,
            num_blocks=xlstm_config.num_blocks,
            embedding_dim=xlstm_config.embedding_dim,
            slstm_at=slstm_at,
        )

        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.out_norm = nn.Identity()

    def forward(self, x):
        x = self.xlstm_stack(x)
        x = self.out_norm(x)
        return x
    
    def reset_parameters(self):
        self.xlstm_stack.reset_parameters()

        
        