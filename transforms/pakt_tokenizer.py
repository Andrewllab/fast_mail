from __future__ import annotations

import logging
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


class PaktTokenizer(Transform, nn.Module):
    """
    Tokenizer that builds tokens by combining multiple embedding branches.

    Adjustments vs your original:
      - Fix token_type_embedding size (3 token types: target/tool/gripper)
      - Fix timestep broadcast (unsqueeze to match (B,T,N,D))
      - Pre-normalize each embedding branch (pos/features/color/time/id/type) BEFORE summation
      - Post-normalize after summation (+ optional linear mixer)
      - Optional feature-branch dropout (regularizes DINO branch without making attention noisy)
      - Robust gather for non-nested tool-action indexing (B,N_src,D) -> (B,T,N,D)
      - A couple broadcast fixes for gripper id/type embeddings
    """

    def __init__(
        self,
        specs: DataSpecs,
        embed_dim: int,
        feature_encoder: Callable[[], nn.Module],
        pos_encoder: Callable[[], nn.Module],
        color_encoder: Optional[Callable[[], nn.Module]] = None,
        timestep_encoder: Optional[Callable[[], nn.Module]] = None,
        num_timesteps: int = 15,
        cartesian_dim: int = 3,  # kept for API compatibility; not used for token types
        num_gripper_points: int = 5,
        feature_dropout_p: float = 0.1,
        use_token_mixer: bool = True,
        separate_obs_action_norms: bool = False,
    ):
        super().__init__()

        self._input_spec = specs.obs

        # Encoders (as provided by your config)
        self.feature_encoder = feature_encoder()
        self.color_encoder = color_encoder() if color_encoder is not None else None
        self.pos_encoder = pos_encoder()

        # Embeddings
        self.gripper_points_id_embedding = nn.Embedding(10, embed_dim)

        # IMPORTANT: token types are (target=0, tool=1, gripper=2, des_gripper=3) => 4 types, not cartesian_dim
        self.token_type_embedding = nn.Embedding(10, embed_dim)

        # +1 because we predict future num_timesteps but also condition on current timestep
        if timestep_encoder is not None:
            self.timestep_encoder = timestep_encoder()
        else:
            self.timestep_embedding = nn.Embedding(num_timesteps + 1, embed_dim)

        # -------- Normalization & mixing (key stability improvements) --------
        # Branch-wise LayerNorm BEFORE summation
        self.pos_ln = nn.LayerNorm(embed_dim)
        self.feature_ln = nn.LayerNorm(embed_dim)
        self.color_ln = (
            nn.LayerNorm(embed_dim) if self.color_encoder is not None else None
        )
        self.time_ln = nn.LayerNorm(embed_dim)
        self.id_ln = nn.LayerNorm(embed_dim)
        self.type_ln = nn.LayerNorm(embed_dim)

        # Feature branch dropout (usually helps reduce jitter without harming conditioning)
        self.feature_dropout = (
            nn.Dropout(p=feature_dropout_p)
            if feature_dropout_p and feature_dropout_p > 0
            else nn.Identity()
        )

        # Post-sum LayerNorm (and optional linear re-mix)
        self.use_token_mixer = use_token_mixer

        if separate_obs_action_norms:
            self.token_ln_obs = nn.LayerNorm(embed_dim)
            self.token_ln_action = nn.LayerNorm(embed_dim)
            self.token_mix_obs = (
                nn.Linear(embed_dim, embed_dim, bias=False)
                if use_token_mixer
                else nn.Identity()
            )
            self.token_mix_action = (
                nn.Linear(embed_dim, embed_dim, bias=False)
                if use_token_mixer
                else nn.Identity()
            )
        else:
            self.token_ln = nn.LayerNorm(embed_dim)
            self.token_mix = (
                nn.Linear(embed_dim, embed_dim, bias=False)
                if use_token_mixer
                else nn.Identity()
            )
            self.token_ln_obs = None
            self.token_ln_action = None
            self.token_mix_obs = None
            self.token_mix_action = None

        self.separate_obs_action_norms = separate_obs_action_norms

        # Define output specs
        obs_embed_spec = EmbedSpec(
            embed_dim=embed_dim,
            n_tokens=None,
            fixed_shape=False,
        )
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        obs_specs["embed"] = obs_embed_spec

        self._output_specs = specs.replace(obs=obs_specs)

        # Store parameters
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps
        self.num_gripper_points = num_gripper_points

        # Store gains used to scale the embedding sums
        def _gain():
            return nn.Parameter(torch.ones(1, embed_dim))

        # Observation/conditioning tokens
        self.g_pos_obs = _gain()
        self.g_feat_obs = _gain()
        self.g_time_obs = _gain()
        self.g_type_obs = _gain()
        self.g_id_obs = _gain()  # only used for gripper-id branch
        self.g_col_obs = _gain() if self.color_encoder is not None else None

        # Action/prediction tokens
        self.g_pos_act = _gain()
        self.g_feat_act = _gain()
        self.g_time_act = _gain()
        self.g_type_act = _gain()
        self.g_id_act = _gain()
        self.g_col_act = _gain() if self.color_encoder is not None else None
        # fmt: on

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

    def get_gains(self, for_action: bool = False) -> dict[str, Tensor]:
        gains = {
            "pos": self.g_pos_act if for_action else self.g_pos_obs,
            "feat": self.g_feat_act if for_action else self.g_feat_obs,
            "time": self.g_time_act if for_action else self.g_time_obs,
            "type": self.g_type_act if for_action else self.g_type_obs,
            "id": self.g_id_act if for_action else self.g_id_obs,
        }
        if self.color_encoder is not None:
            gains["col"] = self.g_col_act if for_action else self.g_col_obs
        return gains

    # ----------------- helpers to support nested/jagged tensors -----------------

    @staticmethod
    def _is_nested(x: Tensor) -> bool:
        return hasattr(x, "is_nested") and bool(x.is_nested)

    def _post_mix_norm(self, x: Tensor, *, is_action: bool) -> Tensor:
        if self.separate_obs_action_norms:
            if is_action:
                x = self.token_mix_action(x)
                x = self.token_ln_action(x)
            else:
                x = self.token_mix_obs(x)
                x = self.token_ln_obs(x)
            return x

        x = self.token_mix(x)
        x = self.token_ln(x)
        return x

    # ----------------- Helpers for different configs -----------------
    def _encode_timestep(self, timesteps: Tensor) -> Tensor:
        if hasattr(self, "timestep_encoder"):
            cur_timesteps = timesteps.to(dtype=torch.float32) / float(
                self.num_timesteps
            )  # normalize to [0,1]
            return self.timestep_encoder(cur_timesteps)
        else:
            return self.timestep_embedding(timesteps)

    # ----------------- tokenizers -----------------
    def __tokenize_pcd(self, pcd: TensorDict, is_action: bool) -> Tensor:
        pos = pcd["points"]  # (B, N_c, 3)
        color = pcd["colors"]  # (B, N_c, 3)
        features = pcd["features"]  # (B, N_c, F)
        # (B, N_c) - not used directly but can be for gripper token id embedding
        gripper_ids = pcd["gripper_ids"]
        # (B, N_c) - not used directly but can be for future extensions (e.g. object-centric token types or ID embeddings)
        object_ids = pcd["object_ids"]
        timesteps = pcd["timesteps"]  # (B, N_c)

        gains = self.get_gains(for_action=is_action)

        # Only gripper points have valid gripper IDs (assuming 0 means non-gripper)
        gripper_mask = gripper_ids.values() > 0
        non_gripper_mask = ~gripper_mask

        # Position embedding + gain
        pos_embed = self.pos_encoder(pos.values())
        pos_embed = self.pos_ln(pos_embed)
        # apply gain here for better stability (before any potential jagged compatibility adjustments)
        pos_embed = pos_embed * gains["pos"]

        # Features
        masked_features = features.values()[non_gripper_mask]  # (B, N_non_grip, F)
        feature_embed = self.feature_encoder(masked_features)  # (B, N_non_grip, D)
        feature_embed = self.feature_ln(feature_embed)
        feature_embed = self.feature_dropout(feature_embed)
        # apply gain here for better stability (before any potential jagged compatibility adjustments)
        feature_embed = feature_embed * gains["feat"]
        pos_embed[non_gripper_mask] += feature_embed

        # Color (optional)
        if self.color_encoder is not None:
            masked_colors = color.values()[non_gripper_mask]  # (B, N_non_grip, 3)
            color_embed = self.color_encoder(masked_colors)
            color_embed = self.color_ln(color_embed)
            color_embed = color_embed * gains["col"]
            pos_embed[non_gripper_mask] += color_embed

        # Object ID embedding
        object_id_embed = self.token_type_embedding(object_ids.values())  # (B, N_c, D)
        object_id_embed = self.type_ln(object_id_embed)
        object_id_embed = object_id_embed * gains["type"]
        pos_embed += object_id_embed

        # Gripper ID embedding (only for gripper points)
        masked_gripper_ids = gripper_ids.values()[gripper_mask]  # (B, N_grip)
        # Using gripper point ID embedding for gripper ids as a simple example
        gripper_ids_embed = self.gripper_points_id_embedding(masked_gripper_ids)
        gripper_ids_embed = self.id_ln(gripper_ids_embed)
        gripper_ids_embed = gripper_ids_embed * gains["id"]
        pos_embed[gripper_mask] += gripper_ids_embed

        # Timestep embedding
        timestep_embed = self._encode_timestep(timesteps.values())  # (B, N_c, D)
        timestep_embed = self.time_ln(timestep_embed)
        timestep_embed = timestep_embed * gains["time"]
        pos_embed += timestep_embed

        token_embed = torch.nested.nested_tensor_from_jagged(
            pos_embed, offsets=pos.offsets()
        )

        token_embed = self._post_mix_norm(token_embed, is_action=is_action)
        return token_embed

    # ----------------- forward -----------------
    def forward(self, batch: TensorDict) -> Tuple[TensorDict, Tensor]:
        obs = batch["obs"]
        actions = batch["noisy_action"]  # shape depends on your pipeline

        # === current obs tokens ===
        current_points = obs["current_points"]
        current_points_tokens = self.__tokenize_pcd(current_points, is_action=False)
        # (B, N_c, D) - not used directly but can be for debugging/visualization

        # === desired future point tokens ===
        future_points = obs["action_points"]
        future_points["points"] = actions
        future_points_tokens = self.__tokenize_pcd(future_points, is_action=True)
        # (B, T*N, D) - not used directly but can be for debugging/visualization

        batch["obs"]["embed"] = current_points_tokens
        return batch, future_points_tokens
