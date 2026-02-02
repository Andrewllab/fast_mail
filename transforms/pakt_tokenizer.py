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
        self.gripper_points_id_embedding = nn.Embedding(num_gripper_points, embed_dim)

        # IMPORTANT: token types are (target=0, tool=1, gripper=2, des_gripper=3) => 4 types, not cartesian_dim
        self.token_type_embedding = nn.Embedding(4, embed_dim)

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

        # Store semi-permanent tensors
        # fmt: off
        self.register_buffer("_tok_target",   torch.tensor([0], dtype=torch.long), persistent=False)
        self.register_buffer("_tok_tool",     torch.tensor([1], dtype=torch.long), persistent=False)
        self.register_buffer("_tok_grip",     torch.tensor([2], dtype=torch.long), persistent=False)
        self.register_buffer("_tok_des_grip", torch.tensor([3], dtype=torch.long), persistent=False)
        
        self.register_buffer("_gripper_point_ids", torch.arange(num_gripper_points, dtype=torch.long), persistent=False)
        self.register_buffer("_zero_timestep", torch.zeros(1, dtype=torch.long), persistent=False)

        # Store gains used to scale the embedding sums
        def _gain():
            return nn.Parameter(torch.ones(1, 1, embed_dim))

        # Observation/conditioning tokens
        self.g_pos_obs  = _gain()
        self.g_feat_obs = _gain()
        self.g_time_obs = _gain()
        self.g_type_obs = _gain()
        self.g_id_obs   = _gain()  # only used for gripper-id branch
        self.g_col_obs  = _gain() if self.color_encoder is not None else None

        # Action/prediction tokens
        self.g_pos_act  = _gain()
        self.g_feat_act = _gain()
        self.g_time_act = _gain()
        self.g_type_act = _gain()
        self.g_id_act   = _gain()
        self.g_col_act  = _gain() if self.color_encoder is not None else None
        # fmt: on

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

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

    def __tokenize_pointcloud(
        self,
        point_pos: Tensor,
        point_color: Tensor,
        point_features: Tensor,
        token_type: Tensor,
    ) -> Tensor:
        """
        Tokenize a point cloud into a set of embedding tokens.

        Args:
            point_pos:      (B, N, 3) or nested equivalent
            point_color:    (B, N, 3) or nested equivalent
            point_features: (B, N, F) or nested equivalent
            token_type:     scalar tensor [0|1|2] (target/tool/gripper)

        Returns:
            (B, N, D) or nested equivalent
        """
        # Encode position
        pos_embed = self.pos_encoder(point_pos)  # (B, N_t, D) -> (B, N_t, D)
        pos_embed = self.pos_ln(pos_embed)

        # Encode features
        feature_embed = self.feature_encoder(
            point_features
        )  # (B, N_t, F) -> (B, N_t, D)
        feature_embed = self.feature_ln(feature_embed)
        feature_embed = self.feature_dropout(feature_embed)
        pos_embed, feature_embed = make_jagged_nested_tensors_compatible(
            pos_embed, feature_embed
        )

        # Optional color branch
        color_embed = None
        if self.color_encoder is not None:
            color_embed = self.color_encoder(point_color)  # (B, N_t, 3) -> (B, N_t, D)
            color_embed = self.color_ln(color_embed)

            # Make jagged tensors compatible if needed
            pos_embed, color_embed = make_jagged_nested_tensors_compatible(
                pos_embed, color_embed
            )

        # Token type embedding (normalize it too)
        token_type_embed = self.token_type_embedding(token_type)  # (1,D)
        token_type_embed = self.type_ln(token_type_embed)
        token_type_embed = token_type_embed.view(1, 1, -1)  # (1,1,D)

        # Timestep embedding (current timestep = 0 for obs tokens)
        timestep_embed = self._encode_timestep(self._zero_timestep)  # (1,D)
        timestep_embed = self.time_ln(timestep_embed)
        timestep_embed = timestep_embed.view(1, 1, -1)  # (1,1,D)

        # Apply gains
        pos_embed = pos_embed * self.g_pos_obs
        feature_embed = feature_embed * self.g_feat_obs
        timestep_embed = timestep_embed * self.g_time_obs
        token_type_embed = token_type_embed * self.g_type_obs

        # Sum branches
        token_embed = (
            pos_embed + feature_embed + token_type_embed + timestep_embed
        )  # (B, N, D)
        if color_embed is not None:
            color_embed = color_embed * self.g_col_obs
            token_embed = token_embed + color_embed

        # Post-sum mixing & normalization (key stability improvement)
        token_embed = self._post_mix_norm(token_embed, is_action=False)
        return token_embed

    def __tokenize_gripper_points(
        self,
        point_pos: Tensor,
        token_type: Tensor,
    ) -> Tensor:
        """
        Tokenize current gripper points (conditioning tokens).

        Args:
            point_pos: (B, N, 3)

        Returns:
            (B, N, D)
        """

        pos_embed = self.pos_encoder(point_pos)
        pos_embed = self.pos_ln(pos_embed)  # (B, N, 3) -> (B, N, D)

        gripper_id_embed = self.gripper_points_id_embedding(
            self._gripper_point_ids
        )  # -> (N, D)
        gripper_id_embed = self.id_ln(gripper_id_embed)
        gripper_id_embed = gripper_id_embed.unsqueeze(0)  # (1, N, D)

        token_type_embed = self.token_type_embedding(token_type)  # (1,D)
        token_type_embed = self.type_ln(token_type_embed)
        token_type_embed = token_type_embed.view(1, 1, -1)  # (1,1,D)

        timestep_embed = self._encode_timestep(self._zero_timestep)  # (1,D)
        timestep_embed = self.time_ln(timestep_embed)
        timestep_embed = timestep_embed.view(1, 1, -1)  # (1,1,D)

        # Apply gains
        pos_embed = pos_embed * self.g_pos_obs
        gripper_id_embed = gripper_id_embed * self.g_id_obs
        timestep_embed = timestep_embed * self.g_time_obs
        token_type_embed = token_type_embed * self.g_type_obs

        # Sum branches
        token_embed = (
            pos_embed + gripper_id_embed + token_type_embed + timestep_embed
        )  # (B, N, D)
        token_embed = self._post_mix_norm(token_embed, is_action=False)
        return token_embed

    def __tokenize_gripper_action_points(
        self,
        point_pos: Tensor,
        gripper_point_ids: Tensor,
        timesteps: Tensor,
        token_type: Tensor,
    ) -> Tensor:
        """
        Tokenize gripper action points (future points).

        Args:
            point_pos:          (B, T*N_g, 3)
            gripper_point_ids:  (B, N_g*T)
            timesteps:          (B, T*N_g)
            token_type:         scalar tensor [0|1|2|3] (target/tool/gripper/des_gripper)

        Returns:
            (B, T*N, D)
        """
        pos_embed = self.pos_encoder(point_pos)  # (B, T*N_g, 3) -> (B, T*N_g, D)
        pos_embed = self.pos_ln(pos_embed)

        # Gripper point ID embedding
        gripper_id_embed = self.gripper_points_id_embedding(
            gripper_point_ids
        )  # (B, T*N_g, D)
        gripper_id_embed = self.id_ln(gripper_id_embed)

        # Timestep embedding (broadcast over N)
        timestep_embed = self._encode_timestep(timesteps)  # (B,T*N_g, D)
        timestep_embed = self.time_ln(timestep_embed)

        # Token type embedding
        token_type_embed = self.token_type_embedding(token_type)  # (1,D)
        token_type_embed = self.type_ln(token_type_embed)
        token_type_embed = token_type_embed.view(1, 1, -1)  # (1,1,D)

        # Apply gains
        pos_embed = pos_embed * self.g_pos_act
        gripper_id_embed = gripper_id_embed * self.g_id_act
        timestep_embed = timestep_embed * self.g_time_act
        token_type_embed = token_type_embed * self.g_type_act

        # Sum branches
        token_embed = pos_embed + gripper_id_embed + timestep_embed + token_type_embed
        token_embed = self._post_mix_norm(token_embed, is_action=True)
        return token_embed

    def __tokenize_tool_action_points(
        self,
        point_pos: Tensor,
        point_color: Tensor,
        point_features: Tensor,
        point_ids: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """
        Tokenize tool/env action points (future points).

        Args:
            point_pos:       (B, T*N, 3) (often nested/jagged)
            point_color:     (B, N_src, 3) OR nested equivalent (current obs colors)
            point_features:  (B, N_src, F) OR nested equivalent (current obs features)
            point_ids:       (B, T*N) indices mapping each future point to a source (obs) point
            timesteps:       (B, N*T) timesteps for each future point

        Returns:
            (B, T*N, D) (often nested/jagged)
        """
        pos_embed = self.pos_encoder(point_pos)  # (B, T*N, 3) -> (B, T*N, D)
        pos_embed = self.pos_ln(pos_embed)

        # ------- Feature Embedding Branch -------
        # Encode features/colors from source set (obs), then index by point_ids to align with future points
        feature_embed_src = self.feature_encoder(
            point_features
        )  # (B, N_src, F) -> (B, N_src, D)
        feature_embed_src = self.feature_ln(feature_embed_src)

        # Index source embeddings by point_ids to match each future token
        if self._is_nested(point_ids) or self._is_nested(feature_embed_src):
            # nested_index_fast is assumed to handle jagged indexing for your pipeline
            feature_embed = nested_index_fast(
                point_ids, feature_embed_src
            )  # (B, T*N), (B, N_src, D) -> (B, T*N, D)
        else:
            # Robust non-nested gather:
            index = point_ids.unsqueeze(-1).expand(-1, -1, feature_embed_src.size(-1))
            feature_embed = torch.gather(
                feature_embed_src, dim=1, index=index
            )  # (B, T*N, D), (B, N_src, D) -> (B, T*N, D)

        # Apply feature dropout after indexing for better noise robustness
        feature_embed = self.feature_dropout(feature_embed)

        # Align jagged-ness with pos_embed if needed
        pos_embed, feature_embed = make_jagged_nested_tensors_compatible(
            pos_embed, feature_embed
        )

        # ------- Optional Color Embedding Branch -------
        color_embed_src = None
        if self.color_encoder is not None:
            color_embed_src = self.color_encoder(
                point_color
            )  # (B, N_src, 3) -> (B, N_src, D)
            color_embed_src = self.color_ln(color_embed_src)
            pos_embed, feature_embed_src = make_jagged_nested_tensors_compatible(
                pos_embed, feature_embed_src
            )

            # Index source embeddings by point_ids to match each future token
            if self._is_nested(point_ids) or self._is_nested(color_embed_src):
                color_embed = nested_index_fast(
                    point_ids, color_embed_src
                )  # (B, T*N), (B, N_src, D) -> (B, T*N, D)
            else:
                # Robust non-nested gather:
                index = point_ids.unsqueeze(-1).expand(-1, -1, color_embed_src.size(-1))
                color_embed = torch.gather(
                    color_embed_src, dim=1, index=index
                )  # (B, T*N, D), (B, N_src, D) -> (B, T*N, D)

            # Align jagged-ness with pos_embed if needed
            pos_embed, color_embed = make_jagged_nested_tensors_compatible(
                pos_embed, color_embed
            )

        # ---- Timestep & Token Type Embeddings ----
        timestep_embed = self._encode_timestep(timesteps)  # (B,T*N,D)
        timestep_embed = self.time_ln(timestep_embed)
        pos_embed, timestep_embed = make_jagged_nested_tensors_compatible(
            pos_embed, timestep_embed
        )

        # ---- Token Type Embedding ----
        token_type_embed = self.token_type_embedding(self._tok_tool)  # (1,D)
        token_type_embed = self.type_ln(token_type_embed)
        token_type_embed = token_type_embed.view(1, 1, -1)  # (1,1,D)

        # Apply gains
        pos_embed = pos_embed * self.g_pos_act
        feature_embed = feature_embed * self.g_feat_act
        timestep_embed = timestep_embed * self.g_time_act
        token_type_embed = token_type_embed * self.g_type_act

        token_embed = pos_embed + feature_embed + timestep_embed + token_type_embed
        if color_embed_src is not None:
            color_embed = color_embed * self.g_col_act
            token_embed = token_embed + color_embed
        token_embed = self._post_mix_norm(token_embed, is_action=True)
        return token_embed

    # ----------------- forward -----------------

    def forward(self, batch: TensorDict) -> Tuple[TensorDict, Tensor]:
        obs = batch["obs"]
        actions = batch["noisy_action"]  # shape depends on your pipeline

        # === target point tokens ===
        target_points = obs["target_points"]
        target_points_pos = target_points["points"]  # (B, N_t, 3)
        target_points_color = target_points["colors"]  # (B, N_t, 3)
        target_points_features = target_points["features"]  # (B, N_t, F)

        target_points_tokens = self.__tokenize_pointcloud(
            point_pos=target_points_pos,
            point_color=target_points_color,
            point_features=target_points_features,
            token_type=self._tok_target,
        )  # (B, N_t, D)

        # === tool point tokens ===
        tool_points = obs["tool_points"]
        tool_points_pos = tool_points["points"]  # (B, N_tool, 3)
        tool_points_color = tool_points["colors"]  # (B, N_tool, 3)
        tool_points_features = tool_points["features"]  # (B, N_tool, F)

        tool_points_tokens = self.__tokenize_pointcloud(
            point_pos=tool_points_pos,
            point_color=tool_points_color,
            point_features=tool_points_features,
            token_type=self._tok_tool,
        )  # (B, N_tool, D)

        # === gripper point tokens ===
        gripper_points = obs["gripper_points"]
        gripper_points_pos = gripper_points["points"]  # (B, N_g, 3)

        gripper_points_tokens = self.__tokenize_gripper_points(
            point_pos=gripper_points_pos,
            token_type=self._tok_grip,
        )  # (B, N_g, D)

        # === desired gripper point tokens ===
        des_gripper_points = obs["des_gripper_points"]
        des_gripper_points_pos = des_gripper_points["points"]  # (B, N_g, 3)
        des_gripper_points_tokens = self.__tokenize_gripper_points(
            point_pos=des_gripper_points_pos,
            token_type=self._tok_des_grip,
        )  # (B, N_g, D)

        # === action point tokens ===
        action_points_pos = actions  # your pipeline provides this
        gripper_action_meta = batch["obs"]["robot_action_points"]
        tool_action_meta = batch["obs"]["tool_action_points"]

        # === gripper action point tokens ===
        num_gripper_action_tokens = self.num_gripper_points * self.num_timesteps

        # NOTE: Keeping your original slicing logic intact, since your actual action tensor layout
        # depends on your pipeline (sometimes already flattened). If this is (B, T, N_a, 3),
        # you may want to revisit this extraction.
        gripper_action_pos = torch.stack(
            [x[:num_gripper_action_tokens] for x in action_points_pos.unbind(0)]
        )  # (B, T*N_g, 3)
        gripper_action_timesteps = gripper_action_meta["timesteps"].long()  # (B, T*N_g)
        gripper_action_point_ids = gripper_action_meta["point_ids"].long()  # (B, N_g*T)

        if ():
            raise ValueError(
                f"Gripper action timesteps out of range [1, {self.num_timesteps}] ({gripper_action_timesteps.min()} < 1 or {gripper_action_timesteps.max()} > {self.num_timesteps})"
            )

        gripper_action_tokens = self.__tokenize_gripper_action_points(
            point_pos=gripper_action_pos,
            gripper_point_ids=gripper_action_point_ids,
            timesteps=gripper_action_timesteps,
            token_type=self._tok_des_grip,
        )  # (B, T*N_g, D)

        # === tool action point tokens ===
        tool_action_pos = torch.nested.nested_tensor(
            [x[num_gripper_action_tokens:] for x in action_points_pos.unbind(0)],
            layout=torch.jagged,
        )  # (B, T*N_tool, 3)

        # These are *source* features/colors for indexing via tool_action_point_ids
        tool_action_features = tool_points_features  # (B, N_tool, F)
        tool_action_color = tool_points_color  # (B, N_tool, 3)

        tool_action_timesteps = tool_action_meta["timesteps"].long()  # (B, N_tool*T)
        tool_action_point_ids = tool_action_meta["point_ids"].long()  # (B, N_tool*T)

        # if (
        #     tool_action_timesteps.max() > self.num_timesteps
        #     or tool_action_timesteps.min() < 1
        #     or gripper_action_timesteps.max() > self.num_timesteps
        #     or gripper_action_timesteps.min() < 1
        # ):
        #     pass

        tool_action_tokens = self.__tokenize_tool_action_points(
            point_pos=tool_action_pos,
            point_color=tool_action_color,
            point_features=tool_action_features,
            point_ids=tool_action_point_ids,
            timesteps=tool_action_timesteps,
        )  # (B, T * N_tool, D)

        # Concatenate along token dimension
        obs_tokens = cat_nested(
            [
                target_points_tokens,
                tool_points_tokens,
                gripper_points_tokens,
                des_gripper_points_tokens,
            ],
            dim=1,
        )
        action_tokens = cat_nested([gripper_action_tokens, tool_action_tokens], dim=1)

        batch["obs"]["embed"] = obs_tokens
        return batch, action_tokens
