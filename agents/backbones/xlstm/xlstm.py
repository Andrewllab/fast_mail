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
        self.ln = LayerNorm(embed_dim, bias)


    def forward(self, x):
        x = self.ln(x)
        x = self.xlstm_stack(x)
        x = self.out_norm(x)
        return x
    
    def reset_parameters(self):
        self.xlstm_stack.reset_parameters()

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

    
class Enc_only(nn.Module):
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

        self.encoder = hydra.utils.instantiate(encoder, block_size=block_size)

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

    def forward(self, states, goals=None, uncond: Optional[bool] = False, 
                keep_last_actions: Optional[bool] = False, 
                return_encoder_embedding: Optional[bool] = False):
        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        if self.goal_conditioned:
            if self.training:
                goals = self.mask_cond(goals)
            if uncond:
                goals = torch.zeros_like(goals).to(self.device)
            goal_embed = self.tok_emb(goals)
        
        state_embed = self.tok_emb(states)
        if self.goal_conditioned:
            goal_x = self.drop(goal_embed[:, :self.goal_seq_len, :])

        state_x = self.drop(state_embed)

        action_seq = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)
        input_seq = torch.cat([state_x, action_seq], dim=1)

        # encode the state, goal and latent z into the hidden dim
        encoder_output = self.encoder(input_seq)

        if return_encoder_embedding:
            return encoder_output

        pred_actions = self.action_pred(encoder_output[:, t:, :])

        return pred_actions

        # add main decoder only blocks
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
        
        