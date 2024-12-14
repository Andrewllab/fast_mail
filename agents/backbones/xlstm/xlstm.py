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
            embed_dim: int,
            n_heads: int,
            attn_pdrop: float,
            resid_pdrop: float,
            n_layers: int,
            block_size: int,
            bias: bool = False,
    ):
        super().__init__()
        # self.blocks = nn.ModuleList(
        #     [mLSTMBlock(config) for _ in range(config.num_blocks)]
        # )

        xlstm_cfg = """ 
vocab_size: 50304
mlstm_block:
mlstm:
    conv1d_kernel_size: 4
    qkv_proj_blocksize: 4
    num_heads: 4
slstm_block:
slstm:
    backend: cuda
    num_heads: 4
    conv1d_kernel_size: 4
    bias_init: powerlaw_blockdependent
feedforward:
    proj_factor: 1.3
    act_fn: gelu
context_length: 256
num_blocks: 7
embedding_dim: 128
slstm_at: [1]
"""
        cfg = OmegaConf.create(xlstm_cfg)
        cfg = from_dict(data_class=xLSTMBlockStackConfig, data=OmegaConf.to_container(cfg), config=DaciteConfig(strict=True))
        self.xlstm_stack = xLSTMBlockStack(cfg)

        self.out_norm = nn.Identity()
        self.ln = LayerNorm(embed_dim, bias)


    # if self.config.add_out_norm:
    #     self.out_norm = RMSNorm(
    #         num_features=config.embedding_dim,
    #         eps=config.norm_eps,
    #         use_weight=True,
    #         use_bias=config.use_bias,
    #         force_float32_reductions=config.norm_reduction_force_float32,
    #     )
    # else:
    #     self.out_norm = nn.Identity()
    #     self.ln = LayerNorm(embed_dim, bias)
    
    def forward(
        self, x: torch.Tensor, state: mLSTMStateType | None = None
    ) -> tuple[torch.Tensor, mLSTMStateType]:
        if state is None:
                state = {i: None for i in range(len(self.blocks))}

    # def forward(self, x):
    #     x = self.ln(x)
    #     x = self.xlstm_stack(x)
    #     x = self.out_norm(x)
    #     return x
    


class Decoder_only(nn.Module):
    def __init__(
            self,
            encoder: DictConfig,
            state_dim: int,
            action_dim: int,
            device: str,
            goal_conditioned: bool,
            embed_dim: int,
            embed_pdrob: float,
            goal_seq_len: int,
            obs_seq_len: int,
            action_seq_len: int,
            goal_drop: float = 0.1,
            linear_output: bool = False,
    ):
        super().__init__()

        self.encoder = hydra.utils.instantiate(encoder)
        
        self.device = device
        self.goal_conditioned = goal_conditioned
        if not goal_conditioned:
            goal_seq_len = 0
        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = goal_seq_len + obs_seq_len + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        # get an action embedding
        self.query_embed = nn.Embedding(action_seq_len, embed_dim)

        # action pred module
        if linear_output:
            self.action_pred = nn.Linear(embed_dim, action_dim)
        else:
            self.action_pred = nn.Sequential(
                nn.Linear(embed_dim, 100),
                nn.GELU(),
                nn.Linear(100, self.action_dim)
            )
        # self.action_pred = nn.Linear(embed_dim, action_dim) # less parameters, worse reward
        self.action_pred.to(self.device)

        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, states, goals=None, uncond: Optional[bool] = False, keep_last_actions: Optional[bool] = False):
        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.goal_conditioned:

            if self.training:
                goals = self.mask_cond(goals)
            # we want to use unconditional sampling during clasisfier free guidance
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)

            goal_embed = self.tok_emb(goals)
        
        # embed the states
        state_embed = self.tok_emb(states)
        if self.goal_conditioned:
            goal_x = self.drop(goal_embed[:, :self.goal_seq_len, :])

        state_x = self.drop(state_embed)

        action_seq = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        input_seq = torch.cat([state_x, action_seq], dim=1)

        # add main decoder only blocks

        
        