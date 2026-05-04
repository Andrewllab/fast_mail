from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor

from environments.specs import DataSpecs, EmbedSpec
from transforms.base_transform import Transform
from utils.nested import (
    cat_nested,
    make_jagged_nested_tensors_compatible,
    nested_index_fast,
)

log = logging.getLogger(__name__)


class EpisodeTimestepEncoder(nn.Module):
    """
    Sinusoidal encoding for ordinal episode timesteps {0, 1, ..., T}.
    Expects raw integer indices — no normalization needed.
    """

    def __init__(self, embed_dim: int, num_timesteps: int, learned: bool = False):
        super().__init__()
        self.learned = learned
        if learned:
            self.emb = nn.Embedding(num_timesteps + 1, embed_dim)
        else:
            half = embed_dim // 2
            freqs = torch.exp(
                -math.log(10000) * torch.arange(half).float() / max(half - 1, 1)
            )
            steps = torch.arange(num_timesteps + 1).float()
            table = steps[:, None] * freqs[None, :]  # (T+1, D/2)
            table = torch.cat([table.sin(), table.cos()], dim=-1)  # (T+1, D)
            self.register_buffer("table", table)

    def forward(self, timesteps: Tensor) -> Tensor:
        # timesteps: (N,) integer indices in {0, ..., T}
        if self.learned:
            return self.emb(timesteps)
        return self.table[timesteps]


class PaktTokenizer(Transform, nn.Module):
    """
    Tokenizer that builds tokens by combining multiple embedding branches
    via concatenation + linear projection (instead of weighted summation).

    Changes vs previous version:
      - EpisodeTimestepEncoder replaces DDPMSigmaEncoder for point timesteps
      - All branches are concatenated and projected via a single Linear (4D -> D)
        which allows arbitrary cross-modality interactions without collapse risk
      - Learned missing_feature_token for gripper points that have no DINOv2 features
      - Pre-norm convention in _post_mix_norm (LN -> Linear)
    """

    def __init__(
        self,
        specs: DataSpecs,
        embed_dim: int,
        feature_encoder: Callable[[], nn.Module],
        pos_encoder: Callable[[], nn.Module],
        color_encoder: Optional[Callable[[], nn.Module]] = None,
        num_timesteps: int = 15,
        cartesian_dim: int = 3,
        num_gripper_points: int = 5,
        feature_dropout_p: float = 0.1,
        use_token_mixer: bool = True,
        separate_obs_action_norms: bool = False,
        episode_timestep_learned: bool = False,
    ):
        super().__init__()

        self._input_spec = specs.obs
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps
        self.num_gripper_points = num_gripper_points

        # -------- Branch encoders --------
        self.feature_encoder = feature_encoder()
        self.color_encoder = color_encoder() if color_encoder is not None else None
        self.pos_encoder = pos_encoder()

        # Episode timestep encoder (sinusoidal by default)
        self.timestep_encoder = EpisodeTimestepEncoder(
            embed_dim=embed_dim,
            num_timesteps=num_timesteps,
            learned=episode_timestep_learned,
        )

        # -------- Embeddings --------
        self.gripper_points_id_embedding = nn.Embedding(10, embed_dim)
        self.token_type_embedding = nn.Embedding(10, embed_dim)

        # Learned token for gripper points that have no DINOv2 feature vector
        # Initialized to zero; will diverge from real feature embeddings during training
        self.missing_feature_token = nn.Parameter(torch.empty(embed_dim))
        nn.init.normal_(self.missing_feature_token, std=0.02)

        # -------- Branch-wise LayerNorms (applied before concat) --------
        self.pos_ln = nn.LayerNorm(embed_dim)
        self.feature_ln = nn.LayerNorm(embed_dim)
        self.color_ln = (
            nn.LayerNorm(embed_dim) if self.color_encoder is not None else None
        )
        self.time_ln = nn.LayerNorm(embed_dim)
        self.id_ln = nn.LayerNorm(embed_dim)
        self.type_ln = nn.LayerNorm(embed_dim)

        # Feature branch dropout
        self.feature_dropout = (
            nn.Dropout(p=feature_dropout_p)
            if feature_dropout_p and feature_dropout_p > 0
            else nn.Identity()
        )

        # -------- Concat + project fusion --------
        # Number of branches that always contribute: pos, feature, time, type
        # id branch only applies to gripper points — we still include it in the
        # concat for all points (zero-filled for non-gripper), so the projection
        # input dim is fixed regardless of point type.
        n_branches = 5  # pos, feature, time, type, id
        if self.color_encoder is not None:
            n_branches += 1
        concat_dim = n_branches * embed_dim

        # -------- Post-fusion mixer (pre-norm convention: LN -> Linear) --------
        self.use_token_mixer = use_token_mixer
        self.separate_obs_action_norms = separate_obs_action_norms

        if separate_obs_action_norms:
            self.fusion_obs = nn.Linear(concat_dim, embed_dim, bias=False)
            self.fusion_act = nn.Linear(concat_dim, embed_dim, bias=False)
            self.token_ln_obs = nn.LayerNorm(embed_dim) if use_token_mixer else None
            self.token_ln_act = nn.LayerNorm(embed_dim) if use_token_mixer else None
        else:
            self.fusion = nn.Linear(concat_dim, embed_dim, bias=False)
            self.token_ln = nn.LayerNorm(embed_dim) if use_token_mixer else None
            self.fusion_obs = None
            self.fusion_act = None
            self.token_ln_obs = None
            self.token_ln_act = None

        # -------- Output specs --------
        obs_specs = dict(specs.obs)
        obs_specs["embed"] = EmbedSpec(
            embed_dim=embed_dim, n_tokens=None, fixed_shape=False
        )
        self._output_specs = specs.replace(obs=obs_specs)

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    # -------- Helpers --------

    @staticmethod
    def _is_nested(x: Tensor) -> bool:
        return hasattr(x, "is_nested") and bool(x.is_nested)

    def _fuse_and_mix(self, branches: list[Tensor], *, is_action: bool) -> Tensor:
        """
        Concatenate branch embeddings and project to embed_dim.
        Follows pre-norm convention: project first, then LayerNorm.
        branches: list of (N_flat, D) tensors, all already branch-LN'd
        """
        x = torch.cat(branches, dim=-1)  # (N_flat, n_branches * D)

        if self.separate_obs_action_norms:
            x = self.fusion_act(x) if is_action else self.fusion_obs(x)
            ln = self.token_ln_act if is_action else self.token_ln_obs
        else:
            x = self.fusion(x)
            ln = self.token_ln

        if ln is not None:
            x = ln(x)

        return x

    # -------- Core tokenizer --------

    def __tokenize_pcd(self, pcd: TensorDict, is_action: bool) -> Tensor:
        pos = pcd["points"]  # nested (B, N, 3)
        features = pcd["features"]  # nested (B, N, F)
        gripper_ids = pcd["gripper_ids"]
        object_ids = pcd["object_ids"]
        timesteps = pcd["timesteps"]  # nested (B, N) — integer episode step indices

        N_flat = pos.values().shape[0]
        device = pos.values().device

        gripper_mask = gripper_ids.values() > 0
        non_gripper_mask = ~gripper_mask

        # ---- Position branch ----
        pos_embed = self.pos_encoder(pos.values())  # (N_flat, D)
        pos_embed = self.pos_ln(pos_embed)  # branch LN

        # ---- Feature branch (with learned missing token for gripper points) ----
        feat_embed = torch.zeros(N_flat, self.embed_dim, device=device)

        if non_gripper_mask.any():
            real_feats = self.feature_encoder(features.values()[non_gripper_mask])
            real_feats = self.feature_ln(real_feats)
            real_feats = self.feature_dropout(real_feats)
            feat_embed[non_gripper_mask] = real_feats

        if gripper_mask.any():
            # Learned parameter broadcast to all gripper points
            missing = self.feature_ln(
                self.missing_feature_token.unsqueeze(0).expand(gripper_mask.sum(), -1)
            )
            missing = self.feature_dropout(missing)   # add this
            feat_embed[gripper_mask] = missing

        # ---- Color branch (optional) ----
        color_branches = []
        if self.color_encoder is not None:
            color = pcd["colors"]
            col_embed = self.color_encoder(color.values())  # (N_flat, D)
            col_embed = self.color_ln(col_embed)
            color_branches = [col_embed]

        # ---- Type branch ----
        type_embed = self.token_type_embedding(object_ids.values())  # (N_flat, D)
        type_embed = self.type_ln(type_embed)

        # ---- ID branch (gripper points get their id embedding; others get zeros) ----
        id_embed = torch.zeros(N_flat, self.embed_dim, device=device)
        if gripper_mask.any():
            g_ids = self.gripper_points_id_embedding(gripper_ids.values()[gripper_mask])
            g_ids = self.id_ln(g_ids)
            id_embed[gripper_mask] = g_ids

        # ---- Timestep branch ----
        # timesteps.values() are raw integer episode step indices — no normalization
        time_embed = self.timestep_encoder(timesteps.values())  # (N_flat, D)
        time_embed = self.time_ln(time_embed)

        # ---- Concat + project ----
        branches = [
            pos_embed,
            feat_embed,
            type_embed,
            id_embed,
            time_embed,
        ] + color_branches
        token_flat = self._fuse_and_mix(branches, is_action=is_action)  # (N_flat, D)

        # Re-wrap as nested tensor using original offsets
        token_embed = torch.nested.nested_tensor_from_jagged(
            token_flat, offsets=pos.offsets()
        )
        return token_embed

    # -------- Forward --------

    def forward(self, batch: TensorDict) -> Tuple[TensorDict, Tensor]:
        obs = batch["obs"]

        current_points_tokens = self.__tokenize_pcd(
            obs["current_points"], is_action=False
        )

        future_points = obs["action_points"]
        future_points["points"] = batch["noisy_action"]
        future_points_tokens = self.__tokenize_pcd(future_points, is_action=True)

        batch["obs"]["embed"] = current_points_tokens
        return batch, future_points_tokens
