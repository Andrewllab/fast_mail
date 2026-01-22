from __future__ import annotations

import logging
from typing import Callable, Tuple

import torch
import torch.nn as nn
from tensordict import NonTensorData, TensorDict
from torch import Tensor
from torch_geometric.data import Batch, Data

from environments.specs import DataSpecs, EmbedSpec, PointCloudSpec
from transforms.base_transform import KeyMapping, Transform
from utils.nested import (
    cat_nested,
    make_jagged_nested_tensors_compatible,
    nested_index_fast,
)

log = logging.getLogger(__name__)


class PaktTokenizer(Transform, nn.Module):

    def __init__(
        self,
        specs: DataSpecs,
        embed_dim: int,
        feature_encoder: Callable[[int], nn.Linear],
        pos_encoder: Callable[[int, int], nn.Linear],
        color_encoder: Callable[[int], nn.Linear] = None,
        num_timesteps: int = 15,
        cartesian_dim: int = 3,
        num_gripper_points: int = 5,
    ):
        super().__init__()

        self._input_spec = specs.obs

        self.feature_encoder = feature_encoder()
        self.color_encoder = color_encoder() if color_encoder is not None else None
        self.pos_encoder = pos_encoder()

        self.gripper_points_id_embedding = nn.Embedding(num_gripper_points, embed_dim)
        self.token_type_embedding = nn.Embedding(
            cartesian_dim, embed_dim
        )  # target, tool, gripper
        self.timestep_embedding = nn.Embedding(
            num_timesteps + 1, embed_dim
        )  # +1 because we predict the future num_timesteps but also condition on the current timestep

        # Define output specs
        obs_embed_spec = EmbedSpec(
            embed_dim=embed_dim,
            n_tokens=None,
            fixed_shape=False,
        )
        obs_specs = dict(specs.obs)  # copy obs specs for local modification
        obs_specs["embed"] = obs_embed_spec

        self._output_specs = specs.replace(
            obs=obs_specs,
        )

        # Store parameters
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps
        self.num_gripper_points = num_gripper_points

        # Store semi-permanent tensors
        # fmt: off
        self.register_buffer("_tok_target", torch.tensor([0], dtype=torch.long), persistent=False)
        self.register_buffer("_tok_tool",   torch.tensor([1], dtype=torch.long), persistent=False)
        self.register_buffer("_tok_grip",   torch.tensor([2], dtype=torch.long), persistent=False)
        self.register_buffer("_gripper_point_ids", torch.arange(num_gripper_points, dtype=torch.long), persistent=False)
        # fmt: on

    @property
    def specs(self) -> DataSpecs:
        return self._output_specs

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
            point_pos (Tensor): Point positions, shape (B, N, 3).
            point_color (Tensor): Point colors, shape (B, N, 3).
            point_features (Tensor): Point features, shape (B, N, F).
            token_type (int): Token type identifier (0: target, 1: tool, 2: gripper).

        Returns:
            Tensor: Point cloud tokens, shape (B, N, D).
        """
        # Encode position
        pos_embed = self.pos_encoder(point_pos)

        # TODO: Make color encoder optional
        # Encode color
        if self.color_encoder is not None:
            color_embed = self.color_encoder(point_color)
            _, color_embed = make_jagged_nested_tensors_compatible(
                pos_embed, color_embed
            )

        # Encode features
        feature_embed = self.feature_encoder(point_features)
        _, feature_embed = make_jagged_nested_tensors_compatible(
            pos_embed, feature_embed
        )

        # Add token type embedding
        token_type_embed = self.token_type_embedding(token_type)

        # Sum embeddings
        token_embed = pos_embed
        token_embed += feature_embed
        token_embed += token_type_embed
        if self.color_encoder is not None:
            token_embed += color_embed

        return token_embed

    def __tokenize_gripper_points(
        self,
        point_pos: Tensor,
    ) -> Tensor:
        """
        Tokenize gripper points into a set of embedding tokens.

        Args:
            point_pos (Tensor): Gripper point positions, shape (B, N, 3).
            gripper_point_ids (Tensor): Gripper point IDs, shape (B, N).

        Returns:
            Tensor: Gripper point tokens, shape (B, N, D).
        """
        B, N, _ = point_pos.shape

        # Encode position
        pos_embed = self.pos_encoder(point_pos)

        # Add gripper point ID embedding
        gripper_id_embed = self.gripper_points_id_embedding(self._gripper_point_ids)

        # Add token type embedding for gripper points (token_type=2)
        token_type_embed = self.token_type_embedding(self._tok_grip)

        # Sum embeddings
        token_embed = pos_embed
        token_embed += gripper_id_embed
        token_embed += token_type_embed

        return token_embed

    def __tokenize_gripper_action_points(
        self,
        point_pos: Tensor,
        gripper_point_ids: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """
        Tokenize gripper action points into a set of embedding tokens.

        Args:
            point_pos (Tensor): Gripper action point positions, shape (B, T, N, 3).
            gripper_point_ids (Tensor): Gripper point IDs, shape (B, T, N).
            timesteps (Tensor): Timesteps, shape (B, T).

        Returns:
            Tensor: Gripper action point tokens, shape (B, T, N, D).
        """

        # Encode position
        pos_embed = self.pos_encoder(point_pos)

        # Add gripper point ID embedding
        gripper_id_embed = self.gripper_points_id_embedding(gripper_point_ids)

        # Add timestep embedding
        timestep_embed = self.timestep_embedding(timesteps)

        # Add token type embedding for gripper points (token_type=2)
        token_type_embed = self.token_type_embedding(self._tok_grip)

        # Sum embeddings
        token_embed = pos_embed
        token_embed += gripper_id_embed
        token_embed += timestep_embed
        token_embed += token_type_embed.unsqueeze(0)

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
        Tokenize tool action points into a set of embedding tokens.

        Args:
            point_pos (Tensor): Tool action point positions, shape (B, T, N, 3).
            point_color (Tensor): Tool action point colors, shape (B, T, N, 3).
            point_features (Tensor): Tool action point features, shape (B, T, N, F).
            point_ids (Tensor): Tool action point IDs, shape (B, T, N).
            timesteps (Tensor): Timesteps, shape (B, T).

        Returns:
            Tensor: Tool action point tokens, shape (B, T, N, D).
        """

        # Encode position
        pos_embed = self.pos_encoder(point_pos)

        # Encode color and feature
        feature_embed = self.feature_encoder(point_features)

        color_feat_embed = feature_embed
        if self.color_encoder is not None:
            color_embed = self.color_encoder(point_color)

            color_embed, feature_embed = make_jagged_nested_tensors_compatible(
                color_embed, feature_embed
            )
            color_feat_embed += color_embed

        # Index by point IDs
        color_feat_embed = nested_index_fast(point_ids, color_feat_embed)

        pos_embed, color_feat_embed = make_jagged_nested_tensors_compatible(
            pos_embed, color_feat_embed
        )

        # Add timestep embedding
        timestep_embed = self.timestep_embedding(timesteps)
        pos_embed, timestep_embed = make_jagged_nested_tensors_compatible(
            pos_embed, timestep_embed
        )

        # Add token type embedding for tool points (token_type=1)
        token_type_embed = self.token_type_embedding(self._tok_tool)

        # Sum embeddings
        token_embed = pos_embed
        token_embed += color_feat_embed
        token_embed += timestep_embed
        token_embed += token_type_embed.unsqueeze(0)

        return token_embed

    def forward(self, batch: TensorDict) -> Tuple[TensorDict, Tensor, Tensor]:
        obs = batch["obs"]
        B = batch.batch_size[0]
        actions = batch["noisy_action"]  # (B, T, N_a, 3)
        device = actions.device

        # === target point tokens ===
        target_points = obs["target_points"]
        target_points_pos = target_points["points"]
        target_points_color = target_points["colors"]
        target_points_features = target_points["features"]

        target_points_tokens = self.__tokenize_pointcloud(
            point_pos=target_points_pos,
            point_color=target_points_color,
            point_features=target_points_features,
            token_type=self._tok_target,  # target
        )  # (B, N_t, D)

        # === tool point tokens ===
        tool_points = obs["tool_points"]
        tool_points_pos = tool_points["points"]
        tool_points_color = tool_points["colors"]
        tool_points_features = tool_points["features"]
        tool_points_tokens = self.__tokenize_pointcloud(
            point_pos=tool_points_pos,
            point_color=tool_points_color,
            point_features=tool_points_features,
            token_type=self._tok_tool,  # tool
        )  # (B, N_tool, D)

        # === gripper point tokens ===
        gripper_points = obs["gripper_points"]
        gripper_points_pos = gripper_points["points"]  # (B, N_g, 3)
        # (B, N_g) # should be 5

        gripper_points_tokens = self.__tokenize_gripper_points(
            point_pos=gripper_points_pos,
        )  # (B, N_g, D)

        # === action point tokens ===
        action_points_pos = actions  # (B, T, N_a, 3)
        gripper_action_meta = batch["obs"]["robot_action_points"]
        tool_action_meta = batch["obs"]["tool_action_points"]

        # === gripper actions point tokens ===
        num_gripper_action_tokens = self.num_gripper_points * self.num_timesteps
        gripper_action_pos = torch.stack(
            [x[:num_gripper_action_tokens] for x in action_points_pos.unbind(0)]
        )
        gripper_action_timesteps = gripper_action_meta["timesteps"]
        gripper_action_point_ids = gripper_action_meta["point_ids"]

        gripper_action_tokens = self.__tokenize_gripper_action_points(
            point_pos=gripper_action_pos,
            gripper_point_ids=gripper_action_point_ids,
            timesteps=gripper_action_timesteps,
        )  # (B, T, 5, D)

        # === tool action point tokens ===
        tool_action_pos = torch.nested.nested_tensor(
            [x[num_gripper_action_tokens:] for x in action_points_pos.unbind(0)],
            layout=torch.jagged,
        )  # (B, T, N_tool, 3) # rest are tool actions
        tool_action_features = tool_points_features  # (B, T, N_tool, F)
        tool_action_color = tool_points_color  # (B, T, N_tool, 3)
        tool_action_timesteps = tool_action_meta["timesteps"]
        tool_action_point_ids = tool_action_meta["point_ids"]
        tool_action_tokens = self.__tokenize_tool_action_points(
            point_pos=tool_action_pos,
            point_color=tool_action_color,
            point_features=tool_action_features,
            point_ids=tool_action_point_ids,
            timesteps=tool_action_timesteps,
        )  # (B, T, N_tool, D)

        # concatenate along N dimension of embedding
        # obs_embed: (B, N, D)
        obs_tokens = cat_nested(
            [target_points_tokens, tool_points_tokens, gripper_points_tokens], dim=1
        )
        action_tokens = cat_nested([gripper_action_tokens, tool_action_tokens], dim=1)

        batch["obs"]["embed"] = obs_tokens

        # batch["action_embed"] = action_tokens
        return batch, action_tokens  # , attention_mask
