import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import einops
import math
from typing import Optional
from torch.nn import functional as F
import logging

logger = logging.getLogger(__name__)


with torch.no_grad():
    from mamba_ssm import Mamba2
    from mamba_ssm import Mamba

    x = torch.randn(2, 5, 256).to("cuda")
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=256,  # Model dimension d_model
        d_state=8,  # SSM state expansion factor
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
    ).to("cuda")
    y = model(x)  # warm up first

    model = Mamba2(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=256,  # Model dimension d_model
        d_state=8,  # SSM state expansion factor, typically 64 or 128
        d_conv=4,  # Local convolution width
        expand=2,  # Block expansion factor
    ).to("cuda")
    y = model(x)  # warm up first


class Enc_only(nn.Module):
    """Diffusion model with transformer architecture for state, goal, time and action tokens,
    with a context size of block_size"""

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

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
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

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
                len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
                len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            states,
            goals=None,
            uncond: Optional[bool] = False,
            keep_last_actions: Optional[bool] = False
    ):

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

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)

        position_embeddings = self.pos_emb[:, :(t + self.goal_seq_len + self.action_seq_len - 1), :]
        # note, that the goal states are at the beginning of the sequence since they are available
        # for all states s_1, ..., s_t otherwise the masking would not make sense
        if self.goal_conditioned:
            goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])

        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])

        # state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:, :])
        # # the action get the same position embedding as the related states
        # action_x = self.drop(action_embed + position_embeddings[:, self.goal_seq_len:, :])

        # decode the action sequence with cross attention over the encoder output
        action_seq = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)

        input_seq = torch.cat([state_x, action_seq], dim=1)

        # encode the state, goal and latent z into the hidden dim
        encoder_output = self.encoder(input_seq)

        pred_actions = self.action_pred(encoder_output[:, t:, :])

        return pred_actions

    def get_params(self):
        return self.parameters()


class Mamba_EncDec(nn.Module):

    def __init__(
            self,
            encoder: DictConfig,
            decoder: DictConfig,
            state_dim: int,
            goal_dim: int,
            action_dim: int,
            obs_tokens: int,
            device: str,
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
        self.decoder = hydra.utils.instantiate(decoder)

        self.device = device

        # input embedding stem
        # first we need to define the maximum block size
        # it consists of the goal sequence length plus 1 for the sigma embedding and 2 the obs seq len
        block_size = goal_seq_len + action_seq_len + obs_seq_len + 1
        # the seq_size is a little different since we have state action pairs for every timestep
        seq_size = obs_tokens + action_seq_len

        self.tok_emb = nn.Linear(state_dim, embed_dim)
        self.tok_emb.to(self.device)

        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, embed_dim))
        self.drop = nn.Dropout(embed_pdrob)
        self.drop.to(self.device)

        self.goal_emb = nn.Linear(goal_dim, embed_dim)

        # needed for calssifier guidance learning
        self.cond_mask_prob = goal_drop

        self.action_dim = action_dim
        self.obs_dim = state_dim
        self.embed_dim = embed_dim

        self.block_size = block_size
        self.goal_seq_len = goal_seq_len
        self.obs_seq_len = obs_seq_len
        self.action_seq_len = action_seq_len

        self.obs_tokens = obs_tokens

        # get an action embedding
        self.action_token = nn.Embedding(action_seq_len, embed_dim)
        self.action_noise_token = nn.Embedding(action_seq_len, embed_dim)
        self.state_token = nn.Embedding(obs_tokens, embed_dim)

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

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    # x: torch.Tensor, t: torch.Tensor, s: torch.Tensor, g: torch.Tensor
    # def forward(self, x, t, state, goal):
    def forward(
            self,
            states,
            goals=None,
            return_encoder_embedding=False
    ):

        if len(states.size()) != 3:
            states = states.unsqueeze(0)

        b, t, dim = states.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # embed them into linear representations for the transformer
        state_embed = self.tok_emb(states)
        goal_embed = self.goal_emb(goals)

        position_embeddings = self.pos_emb

        goal_x = self.drop(goal_embed + position_embeddings[:, :self.goal_seq_len, :])
        state_x = self.drop(state_embed + position_embeddings[:, self.goal_seq_len:(self.goal_seq_len + t), :])

        # decode the action sequence with cross attention over the encoder output
        state_token = self.state_token.weight.unsqueeze(0).repeat(b, 1, 1)
        action_seq = self.action_token.weight.unsqueeze(0).repeat(b, 1, 1)
        action_noise_seq = self.action_noise_token.weight.unsqueeze(0).repeat(b, 1, 1)

        # encode the state, goal and latent z into the hidden dim
        input_seq = torch.cat([goal_x, state_x, action_seq], dim=1)
        encoder_output = self.encoder(input_seq)

        if return_encoder_embedding:
            return encoder_output

        # decoder
        act_seq = torch.cat([state_token, action_noise_seq], dim=1)
        decoder_output = self.decoder(act_seq, cond=encoder_output)

        pred_actions = self.action_pred(decoder_output)[:, self.obs_tokens:, :]

        return pred_actions

    def get_params(self):
        return self.parameters()
